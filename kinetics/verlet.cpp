#include "verlet.h"
#include <cmath>
#include <vector>
namespace kinetics {
VerletIntegrator::VerletIntegrator(double dt) noexcept:dt_(dt),half_dt_(.5*dt){}
void VerletIntegrator::step(System& sys) noexcept{
    for(auto& a:sys.atoms)a.position+=a.velocity*dt_+a.force*(half_dt_/a.mass)*dt_;
    if(!sys.constraints.empty())shake_positions(sys);
    for(auto& a:sys.atoms)a.velocity+=a.force*(half_dt_/a.mass);
}
void VerletIntegrator::finish_step(System& sys) noexcept{
    for(size_t i=0;i<sys.size();++i)sys.atoms[i].velocity+=sys.atoms[i].force*(half_dt_/sys.atoms[i].mass);
    if(!sys.constraints.empty())shake_velocities(sys);
}
void VerletIntegrator::shake_positions(System& sys) const noexcept{
    auto& c=sys.constraints;auto& a=sys.atoms;
    for(size_t iter=0;iter<SHAKE_MAX_ITER;++iter){
        double max_err=0;
        for(auto& cn:c){
            Atom& ai=a[cn.i];Atom& aj=a[cn.j];
            Vec3 rij=aj.position-ai.position;double rij2=rij.norm2(),d02=cn.d0*cn.d0,g=rij2-d02;
            double err=std::abs(g)/d02;if(err>max_err)max_err=err;
            if(std::abs(g)>SHAKE_TOLERANCE*d02){
                double inv=1./ai.mass+1./aj.mass,lam=g/(2.*inv*rij2);
                Vec3 corr=lam*rij;ai.position+=corr/ai.mass;aj.position-=corr/aj.mass;
            }
        }
        if(max_err<SHAKE_TOLERANCE)break;
    }
}
void VerletIntegrator::shake_velocities(System& sys) const noexcept{
    auto& c=sys.constraints;auto& a=sys.atoms;
    for(size_t iter=0;iter<100;++iter){
        double max_err=0;
        for(auto& cn:c){
            Atom& ai=a[cn.i];Atom& aj=a[cn.j];
            Vec3 rij=aj.position-ai.position,vij=aj.velocity-ai.velocity;
            double rv=dot(rij,vij);
            if(std::abs(rv)>1e-10){
                double inv=1./ai.mass+1./aj.mass,kap=rv/(inv*rij.norm2());
                Vec3 corr=kap*rij;ai.velocity+=corr/ai.mass;aj.velocity-=corr/aj.mass;
                double err=std::abs(rv)/(rij.norm()*vij.norm()+1e-10);if(err>max_err)max_err=err;
            }
        }
        if(max_err<1e-8)break;
    }
}
void nve_step(System& sys,VerletIntegrator& vi,void(*ff)(System&))noexcept{vi.step(sys);ff(sys);vi.finish_step(sys);}
}
