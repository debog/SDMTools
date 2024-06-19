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

    Real tf; // s
    Real radius_init; // m
    Real sat_ratio;
    Real temperature; // K
    Real e_sat;
    Real solute_mass; // kg

    FILE* in;
    in = fopen("input", "r");
    if (!in) {
        printf("ERROR: no input file!\n");
        return 1;
    }
    fscanf(in, "%lf", &tf);
    fscanf(in, "%lf", &radius_init);
    fscanf(in, "%lf", &sat_ratio);
    fscanf(in, "%lf", &temperature);
    fscanf(in, "%lf", &e_sat);
    fscanf(in, "%lf", &solute_mass);

    printf("Solute mass: %1.4e kg\n", solute_mass);
    printf("Temperature: %1.4e K\n", temperature);
    printf("Saturation pressure: %1.4e Pa\n", e_sat);
    printf("Saturation ratio: %1.4e\n", sat_ratio);

    Real radius = radius_init;
    printf("Initial radius: %1.16e\n", radius);

    Real init_timescale = drsqdt_rhsjac(  radius*radius,
                                          temperature,
                                          e_sat,
                                          solute_mass );
    printf("Initial timescale: %1.4e\n", 1.0 / std::sqrt(init_timescale*init_timescale));

    Real cfl = 0.0009;

    SuperDropletsUtils::TIRK4< SuperDropletsUtils::dRsqdt_RHSFunc,
                               SuperDropletsUtils::dRsqdt_RHSJac,
                               ParticleReal > ti_rk4 { drsqdt_rhsfun, drsqdt_rhsjac,
                                                       tf, sat_ratio, temperature, e_sat, solute_mass,
                                                       cfl };

    SuperDropletsUtils::TIBE < SuperDropletsUtils::dRsqdt_RHSFunc,
                               SuperDropletsUtils::dRsqdt_RHSJac,
                               SuperDropletsUtils::NewtonSolver<SuperDropletsUtils::dRsqdt_RHSFunc,
                                                                SuperDropletsUtils::dRsqdt_RHSJac,
                                                                ParticleReal>,
                               ParticleReal > ti_be { drsqdt_rhsfun, drsqdt_rhsjac, newton_solver,
                                                      tf, sat_ratio, temperature, e_sat, solute_mass,
                                                      cfl };

    SuperDropletsUtils::TICN < SuperDropletsUtils::dRsqdt_RHSFunc,
                               SuperDropletsUtils::dRsqdt_RHSJac,
                               SuperDropletsUtils::NewtonSolver<SuperDropletsUtils::dRsqdt_RHSFunc,
                                                                SuperDropletsUtils::dRsqdt_RHSJac,
                                                                ParticleReal>,
                               ParticleReal > ti_cn { drsqdt_rhsfun, drsqdt_rhsjac, newton_solver,
                                                      tf, sat_ratio, temperature, e_sat, solute_mass,
                                                      cfl };

    SuperDropletsUtils::TIDIRK212 < SuperDropletsUtils::dRsqdt_RHSFunc,
                                    SuperDropletsUtils::dRsqdt_RHSJac,
                                    SuperDropletsUtils::NewtonSolver<SuperDropletsUtils::dRsqdt_RHSFunc,
                                                                     SuperDropletsUtils::dRsqdt_RHSJac,
                                                                     ParticleReal>,
                                    ParticleReal > ti_dirk2 { drsqdt_rhsfun, drsqdt_rhsjac, newton_solver,
                                                              tf, sat_ratio, temperature, e_sat, solute_mass,
                                                              cfl };

    Real r_sq = radius_init * radius_init;
    //ti_rk4(r_sq);
    //ti_be(r_sq);
    //ti_cn(r_sq);
    ti_dirk2(r_sq);
    radius = std::sqrt(r_sq);
    printf("Final radius: %1.16e m\n", radius);

    return 0;
}

