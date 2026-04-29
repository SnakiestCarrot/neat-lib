#pragma once

#include "neat/config.hpp"

namespace neat {

// Applies matching --key value flags from argv onto cfg, overriding whatever
// defaults the caller already set. Unknown flags are silently skipped so that
// --csv and other env-specific flags coexist without conflict.
//
// Supported flags:
//   --seed <uint64>
//   --population-size <uint>
//   --compat-threshold <double>
//   --dropoff-age <uint>
//   --survival-threshold <double>
//   --prob-add-node <double>
//   --prob-add-link <double>
//   --prob-mutate-weight <double>
//   --prob-toggle-enable <double>
//   --c1 <double>
//   --c2 <double>
//   --c3 <double>
//   --parallel-eval / --no-parallel-eval
//   --activation sigmoid|tanh|relu|leaky_relu
void parse_config_args(Config& cfg, int argc, char* argv[]);

} // namespace neat
