import math,sys,os;sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kinetics import VerletIntegrator,System,Atom,Vec3,BondConstraint,NVE_step
K=100.0;M=12.0;R0=1.5;DT=0.001;N=1000;TOL=1e-4
def diatomic()->System:
    s=System();a1=Atom();a1.position=Vec3(0,0,0);a1.velocity=Vec3(0,0,0);a1.mass=M
    a2=Atom();a2.position=Vec3(R0+.2,0,0);a2.velocity=Vec3(0,2,0);a2.mass=M;s.atoms=[a1,a2];return s
def force(sys:System)->None:
    a1,a2=sys.atoms[0],sys.atoms[1];dx=a2.position.x-a1.position.x;dy=a2.position.y-a1.position.y;dz=a2.position.z-a1.position.z;r=math.sqrt(dx*dx+dy*dy+dz*dz)
    if r>1e-10:fx=-K*(r-R0)*dx/r;fy=-K*(r-R0)*dy/r;fz=-K*(r-R0)*dz/r
    else:fx=fy=fz=0
    a1.force=Vec3(-fx,-fy,-fz);a2.force=Vec3(fx,fy,fz);sys.potential_energy=.5*K*(r-R0)**2
def test()->None:
    s=diatomic();vi=VerletIntegrator(DT);force(s);E0=s.total_energy()
    for _ in range(N):NVE_step(s,vi,force)
    dE=abs(s.total_energy()-E0);dEr=dE/abs(E0)
    print(f"NVE: E0={E0:.10f} Ef={s.total_energy():.10f} dE={dE:.6e} ({dEr*100:.6f}%)");assert dEr<TOL
    s2=diatomic();s2.constraints=[BondConstraint(0,1,R0)];vi2=VerletIntegrator(DT);force(s2)
    for _ in range(N):NVE_step(s2,vi2,force)
    a1,a2=s2.atoms[0],s2.atoms[1];r=math.sqrt((a2.position.x-a1.position.x)**2+(a2.position.y-a1.position.y)**2+(a2.position.z-a1.position.z)**2)
    be=abs(r-R0);print(f"SHAKE: r={r:.8f} error={be:.6e}");assert be<1e-6;print("ALL PASSED")
if __name__=="__main__":test()
