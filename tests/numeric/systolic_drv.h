/*
 * Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
 * Author: Muhammad Faudzan Abdullah
 * SPDX-License-Identifier: Apache-2.0
 */

/* Test shim: generated code includes systolic_drv.h, but the numeric tests run
   on the host where the real driver does not exist. Point it at the stub. */
#include "systolic_stub.h"
