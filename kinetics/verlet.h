#ifndef KINETICS_VERLET_H
#define KINETICS_VERLET_H
#include "types.h"
namespace kinetics {
class VerletIntegrator {
public:
    explicit VerletIntegrator(double dt) noexcept;
    void step(System& sys) noexcept;
    void finish_step(System& sys) noexcept;
    double dt() const noexcept{return dt_;}
    void set_dt(double dt) noexcept{dt_=dt;half_dt_=0.5*dt;}
private:
    double dt_,half_dt_;
    static constexpr double SHAKE_TOLERANCE=1.0e-8;
    static constexpr size_t SHAKE_MAX_ITER=1000;
    void shake_positions(System&) const noexcept;
    void shake_velocities(System&) const noexcept;
};
void nve_step(System&,VerletIntegrator&,void(*)(System&)) noexcept;
}
#endif
