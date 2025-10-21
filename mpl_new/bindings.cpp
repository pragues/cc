#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <mpl_basis/waypoint.h>
#include <mpl_collision/map_util.h>
#include <mpl_planner/planner/map_planner.h>
#include <mpl_traj_solver/poly_traj.h>

#include <iostream>

namespace py = pybind11;
using namespace MPL;

// helper parsers
static void parse_vecf2(py::object obj, Vecf<2> &out) {
    py::sequence seq = py::cast<py::sequence>(obj);
    if (seq.size() != 2) throw std::runtime_error("origin must have length 2");
    out(0) = (decimal_t)py::float_(seq[0]);
    out(1) = (decimal_t)py::float_(seq[1]);
}
static void parse_veci2(py::object obj, Veci<2> &out) {
    py::sequence seq = py::cast<py::sequence>(obj);
    if (seq.size() != 2) throw std::runtime_error("dim must have length 2");
    out(0) = (int)py::int_(seq[0]);
    out(1) = (int)py::int_(seq[1]);
}
static void parse_vecf3(py::object obj, Vecf<3> &out) {
    py::sequence seq = py::cast<py::sequence>(obj);
    if (seq.size() != 3) throw std::runtime_error("origin must have length 3");
    out(0) = (decimal_t)py::float_(seq[0]);
    out(1) = (decimal_t)py::float_(seq[1]);
    out(2) = (decimal_t)py::float_(seq[2]);
}
static void parse_veci3(py::object obj, Veci<3> &out) {
    py::sequence seq = py::cast<py::sequence>(obj);
    if (seq.size() != 3) throw std::runtime_error("dim must have length 3");
    out(0) = (int)py::int_(seq[0]);
    out(1) = (int)py::int_(seq[1]);
    out(2) = (int)py::int_(seq[2]);
}

static Tmap build_tmap_from_py(py::object data_obj) {
    Tmap map_data;
    if (py::isinstance<py::buffer>(data_obj)) {
        py::buffer buf = py::reinterpret_borrow<py::buffer>(data_obj);
        py::buffer_info info = buf.request();
        if (info.ndim != 1) throw std::runtime_error("data must be 1-D");
        ssize_t n = info.size;
        map_data.resize(n);
        if (info.format == py::format_descriptor<int8_t>::format()) {
            memcpy(map_data.data(), info.ptr, n * sizeof(int8_t));
            return map_data;
        }
        if (info.itemsize == sizeof(int32_t)) {
            int32_t *p = static_cast<int32_t*>(info.ptr);
            for (ssize_t i = 0; i < n; ++i) map_data[i] = static_cast<signed char>(p[i]);
            return map_data;
        } else if (info.itemsize == sizeof(int64_t)) {
            int64_t *p = static_cast<int64_t*>(info.ptr);
            for (ssize_t i = 0; i < n; ++i) map_data[i] = static_cast<signed char>(p[i]);
            return map_data;
        } else {
            py::sequence seq = py::cast<py::sequence>(data_obj);
            if (static_cast<ssize_t>(seq.size()) != n) throw std::runtime_error("inconsistent data length");
            for (ssize_t i = 0; i < n; ++i) map_data[i] = static_cast<signed char>(py::int_(seq[i]));
            return map_data;
        }
    } else {
        py::sequence seq = py::cast<py::sequence>(data_obj);
        ssize_t n = seq.size();
        map_data.resize(n);
        for (ssize_t i = 0; i < n; ++i) map_data[i] = static_cast<signed char>(py::int_(seq[i]));
        return map_data;
    }
}

// Vecf<2> helpers (unchanged)
static py::array_t<decimal_t> vecf2_to_ndarray(const Vecf<2> &v) {
    py::array_t<decimal_t> arr({2});
    auto buf = arr.mutable_data();
    buf[0] = v(0);
    buf[1] = v(1);
    return arr;
}
static void ndarray_to_vecf2(const py::object &o, Vecf<2> &v) {
    py::sequence seq = py::cast<py::sequence>(o);
    if (seq.size() != 2) throw std::runtime_error("vec2 must have length 2");
    v(0) = (decimal_t) py::float_(seq[0]);
    v(1) = (decimal_t) py::float_(seq[1]);
}

// Vecf<3> helpers (new)
static py::array_t<decimal_t> vecf3_to_ndarray(const Vecf<3> &v) {
    py::array_t<decimal_t> arr({3});
    auto buf = arr.mutable_data();
    buf[0] = v(0);
    buf[1] = v(1);
    buf[2] = v(2);
    return arr;
}
static void ndarray_to_vecf3(const py::object &o, Vecf<3> &v) {
    py::sequence seq = py::cast<py::sequence>(o);
    if (seq.size() != 3) throw std::runtime_error("vec3 must have length 3");
    v(0) = (decimal_t) py::float_(seq[0]);
    v(1) = (decimal_t) py::float_(seq[1]);
    v(2) = (decimal_t) py::float_(seq[2]);
}

// scalar (yaw) helpers
static decimal_t scalar_from_py(const py::object &o) {
    return (decimal_t) py::float_(o);
}
static py::object scalar_to_py(decimal_t s) {
    return py::cast(s);
}

// returns vec_E<VecDf> so it matches PlannerBase::setU signature
template<int D>
static vec_E<VecDf> parse_U(py::iterable seq) {
    vec_E<VecDf> U;
    for (auto item : seq) {
        // accept list/tuple/numpy array
        py::sequence v = py::cast<py::sequence>(item);
        if (static_cast<ssize_t>(v.size()) != D)
            throw std::runtime_error("each u must have length " + std::to_string(D));
        VecDf u(D);               // dynamic-size vector of length D
        u.setZero();              // initialize
        for (int i = 0; i < D; ++i) {
            u(i) = (decimal_t) py::float_(v[i]);
        }
        U.push_back(u);
    }
    return U;
}


PYBIND11_MODULE(map_planner_py, m) {
    m.doc() = "Python bindings for Motion Primitive Library (MPL)";

    // --- Waypoint bindings ---
    // Waypoint2D with safe properties
    py::class_<Waypoint<2>>(m, "Waypoint2D")
        .def(py::init<>())
        .def_property("pos",
            [](const Waypoint<2>& w){ return vecf2_to_ndarray(w.pos); },
            [](Waypoint<2>& w, py::object o){ ndarray_to_vecf2(o, w.pos); })
        .def_property("vel",
            [](const Waypoint<2>& w){ return vecf2_to_ndarray(w.vel); },
            [](Waypoint<2>& w, py::object o){ ndarray_to_vecf2(o, w.vel); })
        .def_property("acc",
            [](const Waypoint<2>& w){ return vecf2_to_ndarray(w.acc); },
            [](Waypoint<2>& w, py::object o){ ndarray_to_vecf2(o, w.acc); })
        .def_property("jrk",
            [](const Waypoint<2>& w){ return vecf2_to_ndarray(w.jrk); },
            [](Waypoint<2>& w, py::object o){ ndarray_to_vecf2(o, w.jrk); })
        .def_readwrite("t", &Waypoint<2>::t)
        .def_property("use_pos",
                    [](const Waypoint<2>& w) { return w.use_pos; },
                    [](Waypoint<2>& w, bool v) { w.use_pos = v; })
        .def_property("use_vel",
                    [](const Waypoint<2>& w) { return w.use_vel; },
                    [](Waypoint<2>& w, bool v) { w.use_vel = v; })
        .def_property("use_acc",
                    [](const Waypoint<2>& w) { return w.use_acc; },
                    [](Waypoint<2>& w, bool v) { w.use_acc = v; })
        .def_property("use_jrk",
                    [](const Waypoint<2>& w) { return w.use_jrk; },
                    [](Waypoint<2>& w, bool v) { w.use_jrk = v; })
        .def_property("use_yaw",
                    [](const Waypoint<2>& w) { return w.use_yaw; },
                    [](Waypoint<2>& w, bool v) { w.use_yaw = v; })
        .def("__repr__", [](const Waypoint<2>& w) {
            return "<Waypoint2D pos=(" + std::to_string(w.pos(0)) + "," +
                std::to_string(w.pos(1)) + ")>";
        });

    // Waypoint3D (correctly using vecf3 and yaw as scalar)
    py::class_<Waypoint<3>>(m, "Waypoint3D")
        .def(py::init<>())
        .def_property("pos",
            [](const Waypoint<3>& w){ return vecf3_to_ndarray(w.pos); },
            [](Waypoint<3>& w, py::object o){ ndarray_to_vecf3(o, w.pos); })
        .def_property("vel",
            [](const Waypoint<3>& w){ return vecf3_to_ndarray(w.vel); },
            [](Waypoint<3>& w, py::object o){ ndarray_to_vecf3(o, w.vel); })
        .def_property("acc",
            [](const Waypoint<3>& w){ return vecf3_to_ndarray(w.acc); },
            [](Waypoint<3>& w, py::object o){ ndarray_to_vecf3(o, w.acc); })
        .def_property("jrk",
            [](const Waypoint<3>& w){ return vecf3_to_ndarray(w.jrk); },
            [](Waypoint<3>& w, py::object o){ ndarray_to_vecf3(o, w.jrk); })
        .def_property("yaw",
            [](const Waypoint<3>& w){ return scalar_to_py(w.yaw); },
            [](Waypoint<3>& w, py::object o){ w.yaw = scalar_from_py(o); })
        .def_readwrite("t", &Waypoint<3>::t)
        .def_property("use_pos",
                    [](const Waypoint<3>& w) { return w.use_pos; },
                    [](Waypoint<3>& w, bool v) { w.use_pos = v; })
        .def_property("use_vel",
                    [](const Waypoint<3>& w) { return w.use_vel; },
                    [](Waypoint<3>& w, bool v) { w.use_vel = v; })
        .def_property("use_acc",
                    [](const Waypoint<3>& w) { return w.use_acc; },
                    [](Waypoint<3>& w, bool v) { w.use_acc = v; })
        .def_property("use_jrk",
                    [](const Waypoint<3>& w) { return w.use_jrk; },
                    [](Waypoint<3>& w, bool v) { w.use_jrk = v; })
        .def_property("use_yaw",
                    [](const Waypoint<3>& w) { return w.use_yaw; },
                    [](Waypoint<3>& w, bool v) { w.use_yaw = v; })
        .def("__repr__", [](const Waypoint<3>& w) {
            return "<Waypoint3D pos=(" + std::to_string(w.pos(0)) + "," +
                std::to_string(w.pos(1)) + "," + std::to_string(w.pos(2)) + ")>";
        });



    // OccMapUtil (2D) — safe setMap
    py::class_<OccMapUtil, std::shared_ptr<OccMapUtil>>(m, "OccMapUtil")
        .def(py::init<>())
        .def("setMap",
             [](OccMapUtil &self,
                py::object origin_obj,
                py::object dim_obj,
                py::object data_obj,
                decimal_t res) {
                 std::cerr << "[pybind] OccMapUtil::setMap called\n";
                 Vecf<2> origin; parse_vecf2(origin_obj, origin);
                 Veci<2> dim; parse_veci2(dim_obj, dim);
                 Tmap map_data = build_tmap_from_py(data_obj);
                 long long expected = 1LL * dim(0) * dim(1);
                 if (expected != static_cast<long long>(map_data.size())) throw std::runtime_error("dim[0]*dim[1] != data length");
                 std::cerr << "[pybind] OccMapUtil::setMap -> calling C++ setMap, data_size=" << map_data.size() << "\n";
                 self.setMap(origin, dim, map_data, res);
             },
             py::arg("origin"), py::arg("dim"), py::arg("data"), py::arg("resolution"))
        .def("freeUnknown", &OccMapUtil::freeUnknown)
        .def("getCloud", (vec_Vec2f (OccMapUtil::*)()) &OccMapUtil::getCloud)
        .def("is_free", [](OccMapUtil& self, const Veci<2>& idx) { return self.isFree(idx); }, py::arg("idx"))
        .def("is_occupied", [](OccMapUtil& self, const Veci<2>& idx) { return self.isOccupied(idx); }, py::arg("idx"));

    // VoxelMapUtil (3D) — safe setMap
    py::class_<VoxelMapUtil, std::shared_ptr<VoxelMapUtil>>(m, "VoxelMapUtil")
        .def(py::init<>())
        .def("setMap",
             [](VoxelMapUtil &self,
                py::object origin_obj,
                py::object dim_obj,
                py::object data_obj,
                decimal_t res) {
                 std::cerr << "[pybind] VoxelMapUtil::setMap called\n";
                 Vecf<3> origin; parse_vecf3(origin_obj, origin);
                 Veci<3> dim; parse_veci3(dim_obj, dim);
                 Tmap map_data = build_tmap_from_py(data_obj);
                 long long expected = 1LL * dim(0) * dim(1) * dim(2);
                 if (expected != static_cast<long long>(map_data.size())) throw std::runtime_error("dim[0]*dim[1]*dim[2] != data length");
                 std::cerr << "[pybind] VoxelMapUtil::setMap -> calling C++ setMap, data_size=" << map_data.size() << "\n";
                 self.setMap(origin, dim, map_data, res);
             },
             py::arg("origin"), py::arg("dim"), py::arg("data"), py::arg("resolution"))
        .def("freeUnknown", &VoxelMapUtil::freeUnknown)
        .def("getCloud", (vec_Vec3f (VoxelMapUtil::*)()) &VoxelMapUtil::getCloud)
        .def("isFree", [](VoxelMapUtil& self, const Veci<3>& idx) { return self.isFree(idx); }, py::arg("idx"))
        .def("isOccupied", [](VoxelMapUtil& self, const Veci<3>& idx) { return self.isOccupied(idx); }, py::arg("idx"));

    // PolyTraj2D / PolyTraj3D
    py::class_<PolyTraj<2>, std::shared_ptr<PolyTraj<2>>>(m, "PolyTraj2D")
        .def(py::init<>())
        .def("get_total_time", &PolyTraj<2>::getTotalTime)
        .def("evaluate", &PolyTraj<2>::evaluate)
        .def("clear", &PolyTraj<2>::clear);

    py::class_<PolyTraj<3>, std::shared_ptr<PolyTraj<3>>>(m, "PolyTraj3D")
        .def(py::init<>())
        .def("get_total_time", &PolyTraj<3>::getTotalTime)
        .def("evaluate", &PolyTraj<3>::evaluate)
        .def("clear", &PolyTraj<3>::clear);

    // MapPlanner2D (snake_case + camelCase)
    py::class_<MapPlanner<2>, std::shared_ptr<MapPlanner<2>>>(m, "MapPlanner2D")
        .def(py::init<bool>(), py::arg("verbose") = false)
        // snake_case
        .def("set_map_util", &MapPlanner<2>::setMapUtil)
        .def("set_vmax", &MapPlanner<2>::setVmax)
        .def("set_amax", &MapPlanner<2>::setAmax)
        .def("set_dt", &MapPlanner<2>::setDt)
        .def("set_u", [](MapPlanner<2> &self, py::iterable seq) { self.setU(parse_U<2>(seq)); })
        .def("plan", &MapPlanner<2>::plan)
        // .def("get_traj", [](MapPlanner<2> &self) {
        //     const auto &traj = self.getTraj();
        //     size_t n = traj.length();               // Trajectory 自身方法
        //     py::array_t<decimal_t> arr(n * 2);      // 先申请一维连续内存
        //     auto buf = arr.mutable_data();
        //     for (size_t i = 0; i < n; ++i) {
        //         const auto &wp = traj.at(i);       // 用 at() 或 getWaypoint()
        //         buf[i*2 + 0] = wp.pos(0);
        //         buf[i*2 + 1] = wp.pos(1);
        //     }
        //     arr.resize({n, 2});                     // reshape 成二维
        //     return arr;
        // })
        .def("get_close_set", &MapPlanner<2>::getCloseSet)
        // camelCase aliases (same underlying C++ methods)
        .def("setMapUtil", &MapPlanner<2>::setMapUtil)
        .def("setVmax", &MapPlanner<2>::setVmax)
        .def("setAmax", &MapPlanner<2>::setAmax)
        .def("setDt", &MapPlanner<2>::setDt)
        .def("setU", [](MapPlanner<2> &self, py::iterable seq) { self.setU(parse_U<2>(seq)); })
        // .def("getTraj", [](MapPlanner<2> &self) {
        //     const auto &traj = self.getTraj();
        //     size_t n = traj.size();
        //     py::array_t<decimal_t> arr({n, 2});
        //     auto buf = arr.mutable_data();
        //     for (size_t i = 0; i < n; ++i) {
        //         const auto &wp = traj[i];
        //         buf[i * 2 + 0] = wp.pos(0);
        //         buf[i * 2 + 1] = wp.pos(1);
        //     }
        //     return arr;
        // })
        .def("getCloseSet", &MapPlanner<2>::getCloseSet);



    // MapPlanner3D (snake_case + camelCase)
    py::class_<MapPlanner<3>, std::shared_ptr<MapPlanner<3>>>(m, "MapPlanner3D")
        .def(py::init<bool>(), py::arg("verbose") = false)
        // snake_case
        .def("set_map_util", &MapPlanner<3>::setMapUtil)
        .def("set_vmax", &MapPlanner<3>::setVmax)
        .def("set_amax", &MapPlanner<3>::setAmax)
        .def("set_dt", &MapPlanner<3>::setDt)
        .def("set_u", [](MapPlanner<3> &self, py::iterable seq) { self.setU(parse_U<3>(seq)); })
        .def("plan", &MapPlanner<3>::plan)
        // .def("get_traj", [](MapPlanner<3> &self) {
        //     const auto &traj = self.getTraj();
        //     size_t n = traj.size();
        //     py::array_t<decimal_t> arr({n, 3});
        //     auto buf = arr.mutable_data();
        //     for (size_t i = 0; i < n; ++i) {
        //         const auto &wp = traj[i];
        //         buf[i * 3 + 0] = wp.pos(0);
        //         buf[i * 3 + 1] = wp.pos(1);
        //         buf[i * 3 + 2] = wp.pos(2);
        //     }
        //     return arr;
        // })
        .def("get_close_set", &MapPlanner<3>::getCloseSet)
        // camelCase aliases
        .def("setMapUtil", &MapPlanner<3>::setMapUtil)
        .def("setVmax", &MapPlanner<3>::setVmax)
        .def("setAmax", &MapPlanner<3>::setAmax)
        .def("setDt", &MapPlanner<3>::setDt)
        .def("setU", [](MapPlanner<3> &self, py::iterable seq) { self.setU(parse_U<3>(seq)); })
        // .def("getTraj", [](MapPlanner<3> &self) {
        //     const auto &traj = self.getTraj();
        //     size_t n = traj.size();
        //     py::array_t<decimal_t> arr({n, 3});
        //     auto buf = arr.mutable_data();
        //     for (size_t i = 0; i < n; ++i) {
        //         const auto &wp = traj[i];
        //         buf[i * 3 + 0] = wp.pos(0);
        //         buf[i * 3 + 1] = wp.pos(1);
        //         buf[i * 3 + 2] = wp.pos(2);
        //     }
        //     return arr;
        // })
        
        .def("getCloseSet", &MapPlanner<3>::getCloseSet);

    m.attr("OccMapPlanner") = m.attr("MapPlanner2D");
    m.attr("VoxelMapPlanner") = m.attr("MapPlanner3D");

    // // --- OccMapPlanner / VoxelMapPlanner aliases (camelCase) ---
    // // Provide Python-friendly aliases that match the original C++-style camelCase
    // py::class_<MapPlanner<2>, std::shared_ptr<MapPlanner<2>>>(m, "OccMapPlanner")
    //     .def(py::init<bool>(), py::arg("verbose") = false)
    //     // camelCase aliases used in existing python scripts
    //     .def("setMapUtil", &MapPlanner<2>::setMapUtil)
    //     .def("setVmax", &MapPlanner<2>::setVmax)
    //     .def("setAmax", &MapPlanner<2>::setAmax)
    //     .def("setDt", &MapPlanner<2>::setDt)
    //     .def("setU",
    //     [](MapPlanner<2> &self, py::iterable seq) { self.setU(parse_U<2>(seq)); })
    //     .def("plan", &MapPlanner<2>::plan)
    //     .def("getTraj", &MapPlanner<2>::getTraj)
    //     .def("getCloseSet", &MapPlanner<2>::getCloseSet);


    // py::class_<MapPlanner<3>, std::shared_ptr<MapPlanner<3>>>(m, "VoxelMapPlanner")
    //     .def(py::init<bool>(), py::arg("verbose") = false)
    //     .def("setMapUtil", &MapPlanner<3>::setMapUtil)
    //     .def("setVmax", &MapPlanner<3>::setVmax)
    //     .def("setAmax", &MapPlanner<3>::setAmax)
    //     .def("setDt", &MapPlanner<3>::setDt)
    //     .def("setU",
    //     [](MapPlanner<3> &self, py::iterable seq) { self.setU(parse_U<3>(seq)); })
    //     .def("plan", &MapPlanner<3>::plan)
    //     .def("getTraj", &MapPlanner<3>::getTraj)
    //     .def("getCloseSet", &MapPlanner<3>::getCloseSet);

    m.attr("__version__") = "1.0.0";
}

