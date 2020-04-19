#include "PID.h"
#include <algorithm>
#include <iostream>
#include <vector>

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  // Set Kp, Ki, Kd to given vals
  params = {Kp_, Ki_, Kd_};

  // Initialize error to 0.0
  errors = {0, 0, 0};

  // Initialize sum
  prev_cte = 0.0;

  // Best Error
  best_err = std::numeric_limits<double>::max();

  // Set count
  count = 0;
}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  // Proportional Error --> current cte
  errors[0] = cte;

  // Integral Error --> sum of all ctes
  errors[1] += cte;

  // Derivative Error --> rate of change of cte
  errors[2] = cte - prev_cte;

  // Update prev_cte
  prev_cte = cte;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  double Kp, Ki, Kd, p_error, i_error, d_error;
  Kp = params[0]; Ki = params[1]; Kd = params[2];
  p_error = errors[0]; i_error = errors[1]; d_error = errors[2];
  return Kp * p_error + Ki * i_error + Kd * d_error;  // TODO: Add your total error calc here!
}
