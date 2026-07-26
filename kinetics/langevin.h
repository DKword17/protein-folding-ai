#ifndef KINETICS_LANGEVIN_H
#define KINETICS_LANGEVIN_H
#include "types.h"
#include <random>
namespace kinetics {
class LangevinThermostat {
public:
    LangevinThermostat(double dt,double temperature,double friction);
    void set_temperature(double T) noexcept;
    void set_friction(double gamma) noexcept;
    double temperature() const noexcept{return temperature_;}
    double friction() const noexcept{return friction_;}
    double dt() const noexcept{return dt_;}
    void step(System& sys) noexcept;
    void seed(unsigned int s) noexcept;
private:
    double dt_,half_dt_,temperature_,friction_,gamma_dt_,c_,sqrt_1mc2_,sigma_factor_;
    std::mt19937_64 rng_;
    static constexpr double KB=0.0019872041;
    void update_coeffs() noexcept;
};
}
#endif
