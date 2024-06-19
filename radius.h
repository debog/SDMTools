#include <cmath>
#include <stdio.h>

using ParticleReal = double;
using Real = double;

constexpr Real PI      = 3.14159265358979323846264338327950288;
constexpr Real Gamma   = 1.4;             // c_p / c_v [-]
constexpr Real p_0     = 1.0e5;                 // reference surface pressure [Pa]
constexpr Real therco  = 2.40e-02;     // Thermal conductivity of air, J/m/s/K
constexpr Real rhor    = 1000.; // Density of water, kg/m3
constexpr Real R_d     = 287.0;    // dry air constant for dry air [J/(kg-K)]
constexpr Real R_v     = 461.505;  // water vapor constant for water vapor [J/(kg-K)]
constexpr Real Cp_d    = 1004.5;   // We have set this so that with qv=0 we get identically gamma = 1.4
constexpr Real L_v     = 2.5e6;    // latent heat of vaporization (J / kg)
constexpr Real diffelq = 2.21e-05;     // Diffusivity of water vapor, m2/s
constexpr Real Rd_on_Rv = R_d/R_v;
constexpr Real ip_0     = 1./p_0;

Real erf_esatw (Real t) {
    Real const a0 = 6.105851;
    Real const a1 = 0.4440316;
    Real const a2 = 0.1430341e-1;
    Real const a3 = 0.2641412e-3;
    Real const a4 = 0.2995057e-5;
    Real const a5 = 0.2031998e-7;
    Real const a6 = 0.6936113e-10;
    Real const a7 = 0.2564861e-13;
    Real const a8 = -0.3704404e-15;

    Real dtt = t-273.16;

    Real esatw;
    if(dtt > -80.0) {
        esatw = a0 + dtt*(a1+dtt*(a2+dtt*(a3+dtt*(a4+dtt*(a5+dtt*(a6+dtt*(a7+a8*dtt)))))));
    }
    else {
        esatw = 2.0*0.01*std::exp(9.550426 - 5723.265/t + 3.53068*std::log(t) - 0.00728332*t);
    }
    return esatw;
}

void erf_qsatw (Real t, Real p, Real &qsatw) {
    Real esatw;
    esatw = erf_esatw(t);
    qsatw = Rd_on_Rv*esatw/std::max(esatw,p-esatw);
}

Real getPgivenRTh(const Real rhotheta, const Real qv = 0.)
{
    return p_0 * std::pow(R_d * rhotheta * (1.0+(R_v/R_d)*qv) * ip_0, Gamma);
}

Real getThgivenRandT(const Real rho, const Real T, const Real rdOcp, const Real qv=0.0)
{
    // p = rho_d * R_d * T_moist
    Real p_loc = rho * R_d * T * (1.0 + R_v/R_d*qv);
    // theta_d = T * (p0/p)^(R_d/C_p)
    return T * std::pow((p_0/p_loc),rdOcp);
}

namespace MaterialNames
{
    const std::string h2o = "H2O";
    const std::string nacl = "NaCl";
}

/*! Class for material properties */
class MaterialProperties {
    public:

        MaterialProperties ( const std::string& a_name );

        ~MaterialProperties () = default;

        inline Real density() const { return m_density; }
        inline const std::string& name() const { return m_name; }
        inline Real term_vel_a() const { return m_a_tv; }
        inline Real term_vel_b() const { return m_b_tv; }

        inline Real coeffCurv() const { return m_coeff_curv; }
        inline Real coeffVPSolute(const MaterialProperties& a_solute) const
        {
            return m_coeff_VP_solute * m_ionization / a_solute.molWeight();
        }
        inline Real molWeight() const { return m_mol_weight; }
        inline Real latHeatVap() const { return m_lat_vap; }
        inline Real thermCond() const { return m_therm_cond; }
        inline Real Rv() const { return m_Rv; }

    protected:

        std::string m_name;       /*!< name */
        Real m_density;    /*!< density */

        /* terminal velocity coeffs */
        Real m_a_tv, m_b_tv;

        /* other coeffs */
        Real m_coeff_curv;
        Real m_coeff_VP_solute;
        Real m_ionization;
        Real m_mol_weight;
        Real m_lat_vap;
        Real m_therm_cond;
        Real m_Rv;
        Real m_mol_diff;

    private:

        void setProperties_H2O();
        void setProperties_NaCl();

};

namespace SuperDropletsUtils
{
    struct dRsqdt_RHSFunc
    {
        ParticleReal a;
        ParticleReal b;
        ParticleReal L;
        ParticleReal K;
        ParticleReal Rv;
        ParticleReal rho_l;
        ParticleReal D;

        ParticleReal operator() (   const ParticleReal a_R_sq,
                                    const ParticleReal a_S,
                                    const ParticleReal a_T,
                                    const ParticleReal a_e_s,
                                    const ParticleReal a_M_s ) const noexcept
        {
            ParticleReal F_k = ( L/(Rv*a_T) - 1.0) * ((L*rho_l) / (K*a_T));
            ParticleReal F_d = (rho_l*Rv*a_T) / (D*a_e_s);

            ParticleReal R_inv = std::exp(-0.5*std::log(a_R_sq));
            ParticleReal R_inv_cubed = R_inv*R_inv*R_inv;

            ParticleReal alpha = 2.0 * (a_S-1.0) / (F_k + F_d);
            ParticleReal retval = alpha;

            ParticleReal beta = -2.0 * (a/a_T) / (F_k + F_d);
            retval += beta*R_inv;

            ParticleReal gamma = 2.0 * b * a_M_s / (F_k + F_d);
            retval += gamma*R_inv_cubed;

            return retval;
        }

    };

    struct dRsqdt_RHSJac
    {
        ParticleReal a;
        ParticleReal b;
        ParticleReal L;
        ParticleReal K;
        ParticleReal Rv;
        ParticleReal rho_l;
        ParticleReal D;

        ParticleReal operator() (   const ParticleReal a_R_sq,
                                    const ParticleReal a_T,
                                    const ParticleReal a_e_s,
                                    const ParticleReal a_M_s ) const noexcept
        {
            ParticleReal F_k = ( L/(Rv*a_T) - 1.0) * ((L*rho_l) / (K*a_T));
            ParticleReal F_d = (rho_l*Rv*a_T) / (D*a_e_s);

            ParticleReal R_inv = 1.0/std::sqrt(a_R_sq);
            ParticleReal R_inv_3 = R_inv*R_inv*R_inv;
            ParticleReal R_inv_5 = R_inv_3*R_inv*R_inv;

            ParticleReal retval = 0.0;

            ParticleReal beta = -2.0 * (a/a_T) / (F_k + F_d);
            retval -= 0.5 * beta*R_inv_3;

            ParticleReal gamma = 2.0 * b * a_M_s / (F_k + F_d);
            retval -= 0.5 * 3.0*gamma*R_inv_5;

            return retval;
        }

    };

    template<typename RHSFunc, typename JacFunc, typename RT>
    struct NewtonSolver
    {
        RHSFunc&  m_rhs;
        JacFunc&  m_jac;

        RT  m_rtol;
        RT  m_atol;
        RT  m_stol;
        int m_maxits;

        void operator()  (  RT&             a_u,
                            RT&             a_r,
                            const RT&       a_mu,
                            const RT&       a_S,
                            const RT&       a_T,
                            const RT&       a_e_s,
                            const RT&       a_M_s,
                            RT&             a_res_norm_a,
                            RT&             a_res_norm_r,
                            bool&           a_converged ) const
        {
            a_converged = false;
            RT res_norm0 = 0.0;

            for (int k = 0; k < m_maxits; k++) {
                RT residual = a_mu * a_u
                              - (   a_r
                                  + m_rhs( a_u, a_S, a_T, a_e_s, a_M_s ) );
                a_res_norm_a = std::sqrt(residual*residual);

                if (k == 0) {
                    if (a_res_norm_a > 0) {
                        res_norm0 = a_res_norm_a;
                    } else {
                        res_norm0 = 1.0;
                    }
                }
                a_res_norm_r = a_res_norm_a / res_norm0;

//                printf("  iter: %3d, norm: %1.4e (abs.), %1.4e (rel.)\n",
//                        k, a_res_norm_a, a_res_norm_r );

                if (a_res_norm_a <= m_atol) {
                    a_converged = true;
                    break;
                }
                if (a_res_norm_r <= m_rtol) {
                    a_converged = true;
                    break;
                }

                RT slope = a_mu - m_jac( a_u, a_T, a_e_s, a_M_s );
                RT du = 0.0;
                du = - residual / slope;

                RT du_norm = std::sqrt(du*du);
                RT u_norm = std::sqrt(a_u*a_u);
                if (du_norm/u_norm <= m_stol) {
                    a_converged = true;
                    break;
                }

                if (a_u + du < 0) {
                  a_u *= 0.99;
                } else {
                  a_u += du;
                }
            }
        }
    };

    template<typename RHSFunc, typename JacFunc, typename RT>
    struct TIRK4
    {
        RHSFunc& m_rhs;
        JacFunc& m_jac;

        RT m_t_final;
        RT m_S;
        RT m_T;
        RT m_e_s;
        RT m_M_s;

        RT m_cfl;

        void operator() ( RT& a_u ) const
        {
            RT cur_time = 0.0;

            while (cur_time < m_t_final) {

                RT tau = m_jac(a_u, m_T, m_e_s, m_M_s);
                RT dt = m_cfl / std::sqrt(tau*tau);
                if ((cur_time + dt) > m_t_final) {
                    dt = m_t_final - cur_time;
                }

                RT u_new = 0.0;
                while (1) {

                    RT u1 = a_u;
                    RT f1 = m_rhs(u1, m_S, m_T, m_e_s, m_M_s);

                    RT u2 = a_u + 0.5*dt*f1;
                    RT f2 = m_rhs(u2, m_S, m_T, m_e_s, m_M_s);

                    RT u3 = a_u + 0.5*dt*f2;
                    RT f3 = m_rhs(u3, m_S, m_T, m_e_s, m_M_s);

                    RT u4 = a_u + 1.0*dt*f3;
                    RT f4 = m_rhs(u4, m_S, m_T, m_e_s, m_M_s);

                    u_new = a_u + dt*(f1+2.0*f2+2.0*f3+f4)/6.0;

                    if (std::isfinite(std::sqrt(u_new))) {
                        break;
                    }
                    dt *= 0.5;
                    if (dt < (1.0e-12*m_t_final)) {
                        break;
                    }
                    if (!std::isfinite(dt)) {
                        break;
                    }
                }

                a_u = u_new;
                cur_time += dt;

                printf( "Time %1.2e, dt = %1.2e, cfl = %1.1e, radius = %1.4e\n",
                        cur_time, dt, dt * std::sqrt(tau*tau), std::sqrt(a_u));
            }
        }

    };

    template<typename RHSFunc, typename JacFunc, typename NewtonSolver, typename RT>
    struct TIBE
    {
        RHSFunc& m_rhs;
        JacFunc& m_jac;
        NewtonSolver& m_newton;

        RT m_t_final;
        RT m_S;
        RT m_T;
        RT m_e_s;
        RT m_M_s;

        RT m_cfl;

        void operator() ( RT& a_u ) const
        {
            RT cur_time = 0.0;

            while (cur_time < m_t_final) {

                RT tau = m_jac(a_u, m_T, m_e_s, m_M_s);
                RT dt = m_cfl / std::sqrt(tau*tau);
                if ((cur_time + dt) > m_t_final) {
                    dt = m_t_final - cur_time;
                }


                RT res_norm_a = DBL_MAX;
                RT res_norm_r = DBL_MAX;
                bool converged = false;

                RT u_new = 0.0;
                while (1) {

                    RT mu = 1.0 / dt;
                    RT rhs = mu * a_u;
                    u_new = a_u;
                    m_newton( u_new, rhs, mu,
                              m_S, m_T, m_e_s, m_M_s,
                              res_norm_a, res_norm_r, converged );

                    if (std::isfinite(std::sqrt(a_u)) && converged) {
                        break;
                    }
                    dt *= 0.5;
                    if (dt < (1.0e-12*m_t_final)) {
                        break;
                    }
                    if (!std::isfinite(dt)) {
                        break;
                    }
                }

                a_u = u_new;
                cur_time += dt;

                printf( "Time %1.2e, dt = %1.2e, cfl = %1.1e, radius = %1.4e\n",
                        cur_time, dt, dt * std::sqrt(tau*tau), std::sqrt(a_u) );
                printf( "    norms = %1.3e (abs), %1.3e (rel), converged = %s\n",
                        res_norm_a, res_norm_r,
                        (converged ? "yes" : "no") );
            }
        }

    };

    template<typename RHSFunc, typename JacFunc, typename NewtonSolver, typename RT>
    struct TICN
    {
        RHSFunc& m_rhs;
        JacFunc& m_jac;
        NewtonSolver& m_newton;

        RT m_t_final;
        RT m_S;
        RT m_T;
        RT m_e_s;
        RT m_M_s;

        RT m_cfl;

        void operator() ( RT& a_u ) const
        {
            RT cur_time = 0.0;

            while (cur_time < m_t_final) {

                RT tau = m_jac(a_u, m_T, m_e_s, m_M_s);
                RT dt = m_cfl / std::sqrt(tau*tau);
                if ((cur_time + dt) > m_t_final) {
                    dt = m_t_final - cur_time;
                }

                RT res_norm_a = DBL_MAX;
                RT res_norm_r = DBL_MAX;
                bool converged = false;

                RT u_new = 0.0;

                while (1) {

                    RT mu = 1.0 / (0.5*dt);

                    RT u1 = a_u;
                    RT f1 = m_rhs(u1, m_S, m_T, m_e_s, m_M_s);

                    RT u2 = u1;
                    RT rhs = mu * (a_u + 0.5*dt*f1);
                    m_newton( u2, rhs, mu,
                              m_S, m_T, m_e_s, m_M_s,
                              res_norm_a, res_norm_r, converged );
                    RT f2 = m_rhs(u2, m_S, m_T, m_e_s, m_M_s);

                    u_new += 0.5 * dt * (f1 + f2);

                    if (std::isfinite(std::sqrt(u_new)) && converged) {
                        break;
                    }
                    dt *= 0.5;
                    if (dt < (1.0e-12*m_t_final)) {
                        break;
                    }
                    if (!std::isfinite(dt)) {
                        break;
                    }
                }

                a_u = u_new;
                cur_time += dt;

                printf( "Time %1.2e, dt = %1.2e, cfl = %1.1e, radius = %1.4e\n",
                        cur_time, dt, dt * std::sqrt(tau*tau), std::sqrt(a_u) );
                printf( "    norms = %1.3e (abs), %1.3e (rel), converged = %s\n",
                        res_norm_a, res_norm_r,
                        (converged ? "yes" : "no") );
            }
        }

    };

    template<typename RHSFunc, typename JacFunc, typename NewtonSolver, typename RT>
    struct TIDIRK212
    {
        RHSFunc& m_rhs;
        JacFunc& m_jac;
        NewtonSolver& m_newton;

        RT m_t_final;
        RT m_S;
        RT m_T;
        RT m_e_s;
        RT m_M_s;

        RT m_cfl;

        void operator() ( RT& a_u ) const
        {
            RT cur_time = 0.0;

            while (cur_time < m_t_final) {

                RT tau = m_jac(a_u, m_T, m_e_s, m_M_s);
                RT dt = m_cfl / std::sqrt(tau*tau);
                if ((cur_time + dt) > m_t_final) {
                    dt = m_t_final - cur_time;
                }

                bool converged = false;
                RT res_norm_a = DBL_MAX;
                RT res_norm_r = DBL_MAX;
                RT u_new = 0.0;

                while (1) {

                    RT mu = 1.0 / dt;

                    converged = true;
                    res_norm_a = 0.0;
                    res_norm_r = 0.0;

                    RT u1 = a_u;
                    {
                        RT rhs = mu * a_u;
                        RT res_norm_a_i = DBL_MAX;
                        RT res_norm_r_i = DBL_MAX;
                        bool converged_i = false;
                        m_newton( u1, rhs, mu,
                                  m_S, m_T, m_e_s, m_M_s,
                                  res_norm_a_i, res_norm_r_i, converged_i );
                        converged = converged && converged_i;
                        res_norm_a = std::max(res_norm_a, res_norm_a_i);
                        res_norm_r = std::max(res_norm_r, res_norm_r_i);
                    }
                    RT f1 = m_rhs(u1, m_S, m_T, m_e_s, m_M_s);

                    RT u2 = u1;
                    {
                        RT rhs = mu * (a_u - dt*f1);
                        RT res_norm_a_i = DBL_MAX;
                        RT res_norm_r_i = DBL_MAX;
                        bool converged_i = false;
                        m_newton( u2, rhs, mu,
                                  m_S, m_T, m_e_s, m_M_s,
                                  res_norm_a_i, res_norm_r_i, converged_i );
                        converged = converged && converged_i;
                        res_norm_a = std::max(res_norm_a, res_norm_a_i);
                        res_norm_r = std::max(res_norm_r, res_norm_r_i);
                    }
                    RT f2 = m_rhs(u2, m_S, m_T, m_e_s, m_M_s);

                    u_new += 0.5 * dt * (f1 + f2);

                    if (std::isfinite(std::sqrt(u_new)) && converged) {
                        break;
                    }
                    dt *= 0.5;
                    if (dt < (1.0e-12*m_t_final)) {
                        break;
                    }
                    if (!std::isfinite(dt)) {
                        break;
                    }
                }

                a_u = u_new;
                cur_time += dt;

                printf( "Time %1.2e, dt = %1.2e, cfl = %1.1e, radius = %1.4e\n",
                        cur_time, dt, dt * std::sqrt(tau*tau), std::sqrt(a_u) );
                printf( "    norms = %1.3e (abs), %1.3e (rel), converged = %s\n",
                        res_norm_a, res_norm_r,
                        (converged ? "yes" : "no") );
            }
        }

    };
}
