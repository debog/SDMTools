#include <stdio.h>
#include <memory>
#include <vector>
#include <cfloat>
#include "radius.h"

MaterialProperties::MaterialProperties ( const std::string& a_name )
{
    m_name = a_name;
    if (a_name == MaterialNames::h2o) {
        setProperties_H2O();
    } else if (a_name == MaterialNames::nacl) {
        setProperties_NaCl();
    } else {
        printf("ERROR: undefined material in MaterialProperties()\n");
    }
}

void MaterialProperties::setProperties_H2O()
{
    m_density = rhor; // ERF_Constants.H

    m_a_tv = 3.778;
    m_b_tv = 0.67;

    m_coeff_curv = 3.3e-07; // m K
    m_coeff_VP_solute = 4.3e-06; // m^3
    m_ionization = 2;
    m_mol_weight = 1.802e-02; // kg mol^-1
    m_lat_vap = L_v; // ERF_Constants.H
    m_Rv = R_v; // ERF_Constants.H
}

void MaterialProperties::setProperties_NaCl()
{
    m_density = 2170.0;

    m_a_tv = DBL_MAX;
    m_b_tv = DBL_MAX;

    m_coeff_curv = DBL_MAX; // m K
    m_coeff_VP_solute = DBL_MAX; // m^3
    m_ionization = 2;
    m_mol_weight = 5.844e-02; //kg mol^-1
    m_lat_vap = DBL_MAX;
    m_Rv = DBL_MAX;
}

int main()
{
    std::unique_ptr<MaterialProperties> m_vapour_mat;
    std::vector<std::unique_ptr<MaterialProperties>> m_aerosol_mat;
    m_vapour_mat = std::make_unique<MaterialProperties>("H2O");
    m_aerosol_mat.push_back(std::make_unique<MaterialProperties>("NaCl"));

    const Real mat_density = m_vapour_mat->density();

    SuperDropletsUtils::dRsqdt_RHSFunc drsqdt_rhsfun{m_vapour_mat->coeffCurv(),
                                                     m_vapour_mat->coeffVPSolute(*m_aerosol_mat[0]),
                                                     m_vapour_mat->latHeatVap(),
                                                     therco, // ERF_Constants.H
                                                     m_vapour_mat->Rv(),
                                                     mat_density,
                                                     diffelq};

    SuperDropletsUtils::dRsqdt_RHSJac drsqdt_rhsjac{m_vapour_mat->coeffCurv(),
                                                    m_vapour_mat->coeffVPSolute(*m_aerosol_mat[0]),
                                                    m_vapour_mat->latHeatVap(),
                                                    therco, // ERF_Constants.H
                                                    m_vapour_mat->Rv(),
                                                    mat_density,
                                                    diffelq};

    SuperDropletsUtils::NewtonSolver< SuperDropletsUtils::dRsqdt_RHSFunc,
                                      SuperDropletsUtils::dRsqdt_RHSJac,
                                      ParticleReal > newton_solver { drsqdt_rhsfun, drsqdt_rhsjac,
                                                                     1.0e-6,1.0e-99,1.0e-12,100 };

    Real a_dt = 0.5; // s
    Real tf = 0.5; // s
//    Real density = 1.2; // kg m^{-3}
//    Real temperature = 280.0; // K
//    Real solute_mass = 3.3510322e-23; // kg
//    Real qv = 0.0; // vapour fraction

//    Real theta = getThgivenRandT(density, temperature, R_d/Cp_d, qv);
//    Real pressure = getPgivenRTh( density*theta, qv );
//    Real pressure_dry = getPgivenRTh( density*theta );
//    Real e_sat = erf_esatw(temperature)*100;
//    Real sat_humidity; erf_qsatw(temperature, pressure/100.0, sat_humidity);
//    Real sat_ratio = qv/sat_humidity;

//    Real radius_init_cubed = solute_mass / ((4.0/3.0)*PI*mat_density);
//    Real radius_init = std::exp((1.0/3.0)*std::log(radius_init_cubed));

      Real radius_init = 1.6748613302369350e-06;
      Real sat_ratio = 9.5491349965581285e-01;
      Real temperature = 2.9181345858904893e+02;
      Real e_sat = 2.1493020629790435e+03;
      Real solute_mass = 6.1815354741336143e-17;

    printf("Solute mass: %1.4e kg\n", solute_mass);
    printf("Temperature: %1.4e K\n", temperature);
//    printf("Pressure: %1.4e Pa, %1.4e Pa (dry), %1.4e (diff)\n",
//            pressure, pressure_dry, pressure - pressure_dry);
//    printf("Theta: %1.4e\n", theta);
    printf("Saturation pressure: %1.4e Pa\n", e_sat);
//    printf("Saturation humidity: %1.4e\n", sat_humidity);
    printf("Saturation ratio: %1.4e\n", sat_ratio);

    Real radius = radius_init;
    printf("Initial radius: %1.16e m\n", radius);

    for (Real t = 0.0; t < tf; t += a_dt) {

      Real r_sq_0 = radius_init * radius_init;

      // Initial guess
      Real ivt_sd = 1.0 / temperature;
      //Real a = 1.0 / (temperature - 35.86);
      //Real b = a * (temperature - 273.15);
      printf("Curvature coeff: %1.1e\n", m_vapour_mat->coeffCurv());
      Real eq_a = m_vapour_mat->coeffCurv() * ivt_sd;
      Real eq_b = m_vapour_mat->coeffVPSolute(*m_aerosol_mat[0]) * solute_mass;
      Real Rc = std::sqrt(eq_b/eq_a);
      Real eq_c = sat_ratio - 1.0;
      Real a = eq_a / eq_c;
      Real b = eq_b / eq_c;
      Real a3 = a*a*a;

      Real r_init = radius_init;
      if ( (sat_ratio > 1.0) && (a3 < b*(27.0/4.0)) ) {
        r_init = 1.0e-3;
      }
      if (r_init < Rc) {
        r_init = Rc;
      }
      printf("  Rc = %1.16e\n", Rc);

      printf("  initial guess: %1.16e\n", r_init);

      Real r_sq = r_init * r_init;

      Real res_norm_a = DBL_MAX, res_norm_r = DBL_MAX;
      bool converged = false;
      newton_solver ( r_sq, r_sq_0,
                      a_dt,
                      sat_ratio, temperature, e_sat, solute_mass,
                      true, true,
                      res_norm_a, res_norm_r,
                      converged );

      if (converged) {

        radius = std::sqrt(r_sq);

      }

      printf( "Time %1.2e, radius = %1.4e, converged=%s (res norm: %1.3e abs, %1.3e res)\n",
              t+a_dt,
              radius,
              (converged ? "yes" : "no"), res_norm_a, res_norm_r );
    }

    printf("Final radius: %1.16e m\n", radius);

    return 0;
}

