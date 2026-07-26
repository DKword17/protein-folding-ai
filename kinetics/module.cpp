#include "verlet.h"
#include "langevin.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
namespace py=pybind11;using namespace kinetics;
PYBIND11_MODULE(_kinetics_core,m){
    m.doc()="kinetics MD engine";
    py::class_<Vec3>(m,"Vec3").def(py::init<>()).def(py::init<double,double,double>())
        .def_readwrite("x",&Vec3::x).def_readwrite("y",&Vec3::y).def_readwrite("z",&Vec3::z)
        .def("__repr__",[](const Vec3&v){return"Vec3("+std::to_string(v.x)+","+std::to_string(v.y)+","+std::to_string(v.z)+")";})
        .def("norm",&Vec3::norm).def("norm2",&Vec3::norm2)
        .def("__add__",[](const Vec3&a,const Vec3&b){return a+b;})
        .def("__sub__",[](const Vec3&a,const Vec3&b){return a-b;})
        .def("__mul__",[](const Vec3&v,double s){return v*s;})
        .def("__rmul__",[](const Vec3&v,double s){return s*v;})
        .def("__truediv__",[](const Vec3&v,double s){return v/s;})
        .def("dot",&dot);
    py::class_<Atom>(m,"Atom").def(py::init<>())
        .def_readwrite("position",&Atom::position).def_readwrite("velocity",&Atom::velocity)
        .def_readwrite("force",&Atom::force).def_readwrite("mass",&Atom::mass);
    py::class_<BondConstraint>(m,"BondConstraint").def(py::init<>()).def(py::init<size_t,size_t,double>(),py::arg("i"),py::arg("j"),py::arg("d0"))
        .def_readwrite("i",&BondConstraint::i).def_readwrite("j",&BondConstraint::j).def_readwrite("d0",&BondConstraint::d0);
    py::class_<System>(m,"System").def(py::init<>())
        .def_readwrite("atoms",&System::atoms).def_readwrite("constraints",&System::constraints)
        .def_readwrite("potential_energy",&System::potential_energy)
        .def("size",&System::size).def("kinetic_energy",&System::kinetic_energy).def("total_energy",&System::total_energy);
    py::class_<VerletIntegrator>(m,"VerletIntegrator").def(py::init<double>(),py::arg("dt"))
        .def("step",&VerletIntegrator::step,py::arg("sys")).def("finish_step",&VerletIntegrator::finish_step,py::arg("sys"))
        .def_property("dt",&VerletIntegrator::dt,&VerletIntegrator::set_dt);
    py::class_<LangevinThermostat>(m,"LangevinThermostat").def(py::init<double,double,double>(),py::arg("dt"),py::arg("temperature"),py::arg("friction"))
        .def("step",&LangevinThermostat::step,py::arg("sys")).def("set_temperature",&LangevinThermostat::set_temperature,py::arg("T"))
        .def("set_friction",&LangevinThermostat::set_friction,py::arg("gamma")).def("seed",&LangevinThermostat::seed,py::arg("s"))
        .def_property_readonly("temperature",&LangevinThermostat::temperature)
        .def_property_readonly("friction",&LangevinThermostat::friction).def_property_readonly("dt",&LangevinThermostat::dt);
    m.def("NVE_step",[](System&sys,VerletIntegrator&vi,py::function ff){
        vi.step(sys);{py::gil_scoped_acquire g;ff(&sys);}vi.finish_step(sys);
    },py::arg("sys"),py::arg("integrator"),py::arg("force_func"));
}
