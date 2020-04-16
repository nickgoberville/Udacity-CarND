/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  //create random number generator
  std::default_random_engine gen;
  //get normal distributions from x, y, theta
  std::normal_distribution<float> dist_x(x, std[0]);
  std::normal_distribution<float> dist_y(y, std[1]);
  std::normal_distribution<float> dist_theta(theta, std[2]);
  //loop to get random sample from each distribution
  for (int i=0; i<num_particles; ++i){
    Particle n;
    n.id = i;
    n.x = dist_x(gen);
    n.y = dist_y(gen);
    n.theta = dist_theta(gen);
    n.weight = 1.0;
    particles.push_back(n);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  //get random value generator
  std::default_random_engine gen;
  //get gaussian noise distribution from x, y, theta
  std::normal_distribution<float> dist_x(0, std_pos[0]);
  std::normal_distribution<float> dist_y(0, std_pos[1]);
  std::normal_distribution<float> dist_theta(0, std_pos[2]);  
  
  //predict next time-steps x, y, theta for each particle
  for (int i=0; i<num_particles; ++i){
    double theta = particles[i].theta;

    //if yaw_rate doesn't change
    if (fabs(yaw_rate)<0.00001){
      double vel_del = velocity * delta_t;
      particles[i].x += vel_del * cos(theta);
      particles[i].y += vel_del * sin(theta);
    } else{ //if yaw_rate does change
      double vel_yaw = velocity / yaw_rate;
      double yaw_del = yaw_rate * delta_t;
      double theta_yaw_del = theta + yaw_del;
      particles[i].x += vel_yaw * (sin(theta_yaw_del) - sin(theta));
      particles[i].y += vel_yaw * (cos(theta) - cos(theta_yaw_del));
      particles[i].theta += yaw_del;
    }

    //add gaussian noise to x, y, theta
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);   
  }


}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double min_distance, distance;

  for (unsigned int n=0; n<observations.size(); ++n){
    min_distance = 9000000; //initialize min_distance with arbitrary large value   
    int map_id;
    for (unsigned int m=0; m<predicted.size(); ++m){  
      //calculate distance b/t map and observation points
      distance = dist(observations[n].x, observations[n].y, predicted[m].x, predicted[m].y);
      //update map id and min_distance if distance is smaller than min
      if (distance < min_distance){
        map_id = predicted[m].id;
        //observations[n].x = predicted[m].x;
        //observations[n].y = predicted[m].y;
        min_distance = distance;
      }
    }
    observations[n].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double x, y, theta, landmrk_x, landmrk_y, landmrk_dist;  
  int landmrk_id;
  for (int i=0; i<num_particles; ++i){
    x = particles[i].x;
    y = particles[i].y;
    theta = particles[i].theta;

    //setup vector for landmarks within particles range
    vector<LandmarkObs> predicted;
    for (unsigned int j=0; j<map_landmarks.landmark_list.size(); ++j){
      landmrk_x = map_landmarks.landmark_list[j].x_f;
      landmrk_y = map_landmarks.landmark_list[j].y_f;
      landmrk_id = map_landmarks.landmark_list[j].id_i;
      landmrk_dist = dist(x, y, landmrk_x, landmrk_y);
      if (landmrk_dist<=sensor_range){
        predicted.push_back(LandmarkObs{landmrk_id, landmrk_x, landmrk_y});
      }
    }

    vector<LandmarkObs> transformed_observations;
    for (unsigned int j=0; j<observations.size(); ++j){
      double cos_p = cos(theta);
      double sin_p = sin(theta);
      transformed_observations.push_back(LandmarkObs{
        observations[j].id, 
        cos_p * observations[j].x - sin_p * observations[j].y + x,
        sin_p * observations[j].x + cos_p*observations[j].y + y});
    }

    dataAssociation(predicted, transformed_observations);

    particles[i].weight = 1.0;

    for (unsigned int j=0; j<transformed_observations.size(); ++j){
      double obs_x = transformed_observations[j].x;
      double obs_y = transformed_observations[j].y;

        double pred_x, pred_y;

        bool found = false;
        unsigned int k = 0;
        while (!found && k < predicted.size()){
          if (predicted[k].id == transformed_observations[j].id){
            pred_x = predicted[k].x;
            pred_y = predicted[k].y;          
        }
        k++;
        }

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double obs_w = (1/(2*M_PI*std_x*std_y)) * exp(-(pow(pred_x-obs_x,2)/(2*pow(std_x, 2)) + (pow(pred_y-obs_y,2)/(2*pow(std_y, 2)))));

    //update particle weight
    particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  vector<double> weights;
  double max_weight = 0.00000001;
  for (int i=0; i<num_particles; ++i){
    weights.push_back(particles[i].weight);
    if (particles[i].weight > max_weight){
      max_weight = particles[i].weight;
    }
  }

  std::uniform_real_distribution<double> dist_double(0.0, max_weight);
  std::uniform_int_distribution<int> dist_int(0, num_particles-1);

  int index = dist_int(gen);

  double beta = 0.0;

  vector<Particle> resampledParticles;
  for (int i=0; i<num_particles; ++i){
    beta += dist_double(gen) * 2.0;
    while (beta > weights[index]){
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }
  particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}