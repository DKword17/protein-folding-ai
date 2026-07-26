#ifndef KINETICS_TYPES_H
#define KINETICS_TYPES_H
#include <cmath>
#include <cstddef>
#include <vector>
#include <stdexcept>
namespace kinetics {
struct Vec3 {
    double x{0.0}, y{0.0}, z{0.0};
    Vec3() noexcept = default;
    Vec3(double x, double y, double z) noexcept : x(x), y(y), z(z) {}
    Vec3& operator+=(const Vec3& o) noexcept { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o) noexcept { x-=o.x; y-=o.y; z-=o.z; return *this; }
    Vec3& operator*=(double s) noexcept { x*=s; y*=s; z*=s; return *this; }
    Vec3& operator/=(double s) { if(s==0)throw std::runtime_error("div0"); x/=s; y/=s; z/=s; return *this; }
    double norm2() const noexcept { return x*x+y*y+z*z; }
    double norm() const noexcept { return std::sqrt(norm2()); }
};
inline Vec3 operator+(const Vec3& a,const Vec3& b) noexcept{return Vec3(a.x+b.x,a.y+b.y,a.z+b.z);}
inline Vec3 operator-(const Vec3& a,const Vec3& b) noexcept{return Vec3(a.x-b.x,a.y-b.y,a.z-b.z);}
inline Vec3 operator*(const Vec3& v,double s) noexcept{return Vec3(v.x*s,v.y*s,v.z*s);}
inline Vec3 operator*(double s,const Vec3& v) noexcept{return Vec3(v.x*s,v.y*s,v.z*s);}
inline Vec3 operator/(const Vec3& v,double s){if(s==0)throw std::runtime_error("div0");return Vec3(v.x/s,v.y/s,v.z/s);}
inline double dot(const Vec3& a,const Vec3& b) noexcept{return a.x*b.x+a.y*b.y+a.z*b.z;}
struct BondConstraint{size_t i,j;double d0;};
struct Atom{Vec3 position,velocity,force;double mass;};
struct System{
    std::vector<Atom> atoms;
    std::vector<BondConstraint> constraints;
    size_t size() const noexcept{return atoms.size();}
    double kinetic_energy() const noexcept{double Ek=0;for(auto&a:atoms)Ek+=a.mass*a.velocity.norm2();return 0.5*Ek;}
    double potential_energy{0.0};
    double total_energy() const noexcept{return kinetic_energy()+potential_energy;}
};
}
#endif
