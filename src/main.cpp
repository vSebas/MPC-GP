#include "model/usv_dynamic_model.h"

#include <cstdio>

int main() {
  UsvDynamicModel usv;

  // Ask for 40 N surge and 2 Nm yaw for 10 s (1000 steps of 0.01 s)
  auto [T_port, T_stbd] = UsvDynamicModel::allocate(40.0, 2.0);
  std::printf("T_port = %.2f N, T_stbd = %.2f N\n", T_port, T_stbd);

  std::printf("%6s %8s %8s %8s %8s %8s\n", "t", "x", "y", "psi", "u", "r");
  for (int i = 1; i <= 1000; ++i) {
    State s = usv.update(T_port, T_stbd);
    if (i % 100 == 0)
      std::printf("%6.2f %8.3f %8.3f %8.3f %8.3f %8.3f\n", i * 0.01, s.x, s.y,
                  s.psi, s.u, s.r);
  }
  return 0;
}
