#include <pybind11/pybind11.h>

#include "mu/runtime_config_c_api.h"

namespace py = pybind11;

PYBIND11_MODULE(_runtime_config_bindings, m) {
  m.doc() = "TensorFlow MUSA runtime configuration bindings";

  m.def(
      "set_musa_allow_growth",
      [](bool enabled) { TFMusaSetAllowGrowth(enabled ? 1 : 0); },
      py::arg("enabled"));
}
