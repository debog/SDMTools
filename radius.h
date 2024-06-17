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
                                    const ParticleReal a_M_s,
                                    const bool a_include_curv,
                                    const bool a_include_solute ) const noexcept
        {
            ParticleReal F_k = ( L/(Rv*a_T) - 1.0) * ((L*rho_l) / (K*a_T));
            ParticleReal F_d = (rho_l*Rv*a_T) / (D*a_e_s);

            ParticleReal R_inv = std::exp(-0.5*std::log(a_R_sq));
            ParticleReal R_inv_cubed = R_inv*R_inv*R_inv;

            ParticleReal alpha = 2.0 * (a_S-1.0) / (F_k + F_d);
            ParticleReal retval = alpha;

            if (a_include_curv) {
              ParticleReal beta = -2.0 * (a/a_T) / (F_k + F_d);
              retval += beta*R_inv;
            }

            if (a_include_solute) {
              ParticleReal gamma = 2.0 * b * a_M_s / (F_k + F_d);
              retval += gamma*R_inv_cubed;
            }

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
                                    const ParticleReal a_M_s,
                                    const bool a_include_curv,
                                    const bool a_include_solute ) const noexcept
        {
            ParticleReal F_k = ( L/(Rv*a_T) - 1.0) * ((L*rho_l) / (K*a_T));
            ParticleReal F_d = (rho_l*Rv*a_T) / (D*a_e_s);

            ParticleReal R_inv = 1.0/std::sqrt(a_R_sq);
            ParticleReal R_inv_3 = R_inv*R_inv*R_inv;
            ParticleReal R_inv_5 = R_inv_3*R_inv*R_inv;

            ParticleReal retval = 0.0;

            if (a_include_curv) {
                ParticleReal beta = -2.0 * (a/a_T) / (F_k + F_d);
                retval -= 0.5 * beta*R_inv_3;
            }

            if (a_include_solute) {
                ParticleReal gamma = 2.0 * b * a_M_s / (F_k + F_d);
                retval -= 0.5 * 3.0*gamma*R_inv_5;
            }

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
                            const RT&       a_dt,
                            const RT&       a_S,
                            const RT&       a_T,
                            const RT&       a_e_s,
                            const RT&       a_M_s,
                            const bool      a_incl_curv,
                            const bool      a_incl_sol,
                            RT&             a_res_norm_a,
                            RT&             a_res_norm_r,
                            bool&           a_converged ) const
        {
            a_converged = false;
            RT res_norm0 = 0.0;

            for (int k = 0; k < m_maxits; k++) {
                RT residual = a_u
                              - (   a_r
                                  + a_dt * m_rhs( a_u, a_S, a_T, a_e_s, a_M_s, a_incl_curv, a_incl_sol ));
                a_res_norm_a = std::sqrt(residual*residual);

                if (k == 0) { res_norm0 = a_res_norm_a; }
                a_res_norm_r = a_res_norm_a / res_norm0;

                printf("  iter: %3d, norm: %1.4e (abs.), %1.4e (rel.)\n",
                        k, a_res_norm_a, a_res_norm_r );

                if (a_res_norm_a <= m_atol) {
                    a_converged = true;
                    break;
                }
                if (a_res_norm_r <= m_rtol) {
                    a_converged = true;
                    break;
                }

                RT slope = 1.0 - a_dt * m_jac( a_u, a_T, a_e_s, a_M_s, a_incl_curv, a_incl_sol );
                RT du = 0.0;
//                if ((slope*slope) < (10000*residual*residual)) {
//                    du = 0.0;
//                } else {
                    du = - residual / slope;
//                }

                RT du_norm = std::sqrt(du*du);
                RT u_norm = std::sqrt(a_u*a_u);
                if (du_norm/u_norm <= m_stol) {
                    a_converged = true;
                    break;
                }

                if (a_u + du < 0) {
                  a_u *= 0.99;
                  //a_u += 0.01*a_u;
                } else {
                  a_u += du;
                }
            }
        }
    };
}
