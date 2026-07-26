#include "langevin.h"
#include <cmath>
#include <chrono>
namespace kinetics {
LangevinThermostat::LangevinThermostat(double dt,double T,double gamma)
:dt_(dt),half_dt_(.5*dt),temperature_(T),friction_(gamma),gamma_dt_(gamma*dt),
 c_(std::exp(-gamma_dt_)),rng_(static_cast<unsigned long long>(std::chrono::system_clock::now().time_since_epoch().count())){
    update_coeffs();
}
void LangevinThermostat::set_temperature(double T) noexcept{temperature_=T;update_coeffs();}
void LangevinThermostat::set_friction(double g) noexcept{
    friction_=g;gamma_dt_=g*dt_;c_=std::exp(-gamma_dt_);update_coeffs();
}
void LangevinThermostat::step(System& sys) noexcept{
    for(auto& a:sys.atoms)a.velocity+=a.force*(half_dt_/a.mass);
    for(auto& a:sys.atoms)a.position+=a.velocity*half_dt_;
    for(auto& a:sys.atoms){
        double sigma=sigma_factor_/std::sqrt(a.mass);
        std::normal_distribution<double> n{0.,1.};
        a.velocity.x=c_*a.velocity.x+sqrt_1mc2_*sigma*n(rng_);
        a.velocity.y=c_*a.velocity.y+sqrt_1mc2_*sigma*n(rng_);
        a.velocity.z=c_*a.velocity.z+sqrt_1mc2_*sigma*n(rng_);
    }
    for(auto& a:sys.atoms)a.position+=a.velocity*half_dt_;
    for(auto& a:sys.atoms)a.velocity+=a.force*(half_dt_/a.mass);
}
void LangevinThermostat::seed(unsigned int s) noexcept{rng_.seed(s);}
void LangevinThermostat::update_coeffs() noexcept{
    sigma_factor_=std::sqrt(KB*temperature_);sqrt_1mc2_=std::sqrt(1.-c_*c_);
}
}
