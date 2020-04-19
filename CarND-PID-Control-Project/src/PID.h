#ifndef PID_H
#define PID_H

#include <vector>
#include <ctime>


class PID {
 public:
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();

 private:
  /**
   * PID Errors
   */
  std::vector<double> errors;

  /**
   * PID Coefficients
   */ 
  std::vector<double> params;

  /**
   *  PID Error tools
   */
  double prev_cte;
  double set_point;

  /**
   * Parameters used within Twiddle()
   */
  bool Kp_set;
  bool Ki_set;
  bool Kd_set;
  double best_err;
  int count; 
};

#endif  // PID_H