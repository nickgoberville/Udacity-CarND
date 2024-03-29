#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

struct Vehicle
{
  int id;
  float dist;
  float speed;
  float rel_speed;
  int lane;
  bool warning;
};

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  //start in lane 1
  int lane = 1;


  //have a reference velocity to target
  double ref_vel = 0; //mph

  h.onMessage([&lane, &ref_vel, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];


          /**
           * BEGINNING OF MY CODE
           */ 
          //number of points previously used
          int prev_size = previous_path_x.size();
          
          //lane_offset value based on lane we are in
          int lane_offset = 2+4*lane;
          int left_lane = 2;
          int center_lane = 6;
          int right_lane = 10;
          //setting tolerance for d for checking if vehicle is in lane
          float d_tol = 2;
          //setting tolerance for s for checking if vehicle is in lane
          float s_tol = ref_vel*0.8;
          // changing lane tolerance for s
          float turn_tol = 30;
          //set speed limit
          float speed_limit = 48;

          //sensor fusion code
          if (prev_size > 0)
          {
            car_s = end_path_s;
          }

          bool too_close = false;

          vector<Vehicle> left_cars;
          vector<Vehicle> center_cars;
          vector<Vehicle> right_cars;
          //VERSION 3
          for (int i = 0; i < sensor_fusion.size(); ++i)
          {
            //calculate vehicle distance
            float dist = distance(car_x, car_y, (float)sensor_fusion[i][1], (float)sensor_fusion[i][2]);
            if (dist < turn_tol)
            {
              //get the vehicle's speed
              float car_speed = distance(0, (float)sensor_fusion[i][3], 0, (float)sensor_fusion[i][3])*2.24;
              float car_rel_speed = ref_vel - car_speed;
              //get the lane the detected vehicle is in              
              float d = sensor_fusion[i][6];
              int vehicle_lane;
              if (d < (left_lane+d_tol) && d > (left_lane-d_tol)){
                vehicle_lane = left_lane;
              } else if (d < (center_lane+d_tol) && d > (center_lane-d_tol)){
                vehicle_lane = center_lane;
              } else if (d < (right_lane+d_tol) && d > (right_lane-d_tol)){
                vehicle_lane = right_lane;
              }
              //identify if this is a warning or not
              bool warning = false;
              if ((dist < 30.0))// || (dist < 20 && car_rel_speed > 8))
              {
                bool warning = true;
              }
            if (vehicle_lane == left_lane){
              left_cars.push_back({i, dist, car_speed, car_rel_speed, vehicle_lane, warning});
            } else if (vehicle_lane == right_lane){
              right_cars.push_back({i, dist, car_speed, car_rel_speed, vehicle_lane, warning});
            } else{
              center_cars.push_back({i, dist, car_speed, car_rel_speed, vehicle_lane, warning});
            } 
            }
          }

          for (int i = 0; i < left_cars.size(); ++i)
          {
            //print all values out
            std::cout << "Sensor fusion index: " << left_cars[i].id << std::endl;
            std::cout << "\tdistance: " << left_cars[i].dist << std::endl;
            std::cout << "\tspeed: " << left_cars[i].speed << std::endl;
            std::cout << "\trelative speed: " << left_cars[i].rel_speed << std::endl;
            std::cout << "\tlane: " << "LEFT" << std::endl;
            std::cout << "\twarning: " << left_cars[i].warning << std::endl;
          }
          for (int i = 0; i < center_cars.size(); ++i)
          {
            //print all values out
            std::cout << "Sensor fusion index: " << center_cars[i].id << std::endl;
            std::cout << "\tdistance: " << center_cars[i].dist << std::endl;
            std::cout << "\tspeed: " << center_cars[i].speed << std::endl;
            std::cout << "\trelative speed: " << center_cars[i].rel_speed << std::endl;
            std::cout << "\tlane: " << "CENTER" << std::endl;
            std::cout << "\twarning: " << center_cars[i].warning << std::endl;
          }
          for (int i = 0; i < right_cars.size(); ++i)
          {
            //print all values out
            std::cout << "Sensor fusion index: " << right_cars[i].id << std::endl;
            std::cout << "\tdistance: " << right_cars[i].dist << std::endl;
            std::cout << "\tspeed: " << right_cars[i].speed << std::endl;
            std::cout << "\trelative speed: " << right_cars[i].rel_speed << std::endl;
            std::cout << "\tlane: " << "RIGHT" << std::endl;
            std::cout << "\twarning: " << right_cars[i].warning << std::endl;
          }                    
          
          //for each car detected from sensors
          // VERSION 1 -- SENSOR FUSION
          for (int i = 0; i < sensor_fusion.size(); ++i)
          {
            //check if car is in my lane
            float d = sensor_fusion[i][6];
            // if in same lane
            if (d < (lane_offset+d_tol) && d > (lane_offset-d_tol))
            {
              double vx = sensor_fusion[i][3]; //gets vehicle x velocity
              double vy = sensor_fusion[i][4]; //gets vehicle y velocity
              double check_speed = sqrt(vx*vx+vy*vy); //get magnitude of velocity using x,y vals
              double check_car_s = sensor_fusion[i][5]; //gets distance ahead from vehicle

              check_car_s += ((double)prev_size*.02*check_speed); //if using previous points can project s value out
              //check s values greater than mine and s gap
              if((check_car_s > car_s) && ((check_car_s - car_s) < s_tol))
              {
                // TODO: make this more sophisticated
                // --> Use relative velocity to determine how fast to slow down
                //
                too_close = true;
                speed_limit = check_speed*2.24;
                float relative_vel = ref_vel - speed_limit;
              }
            }
          }

          if (too_close)
          {
            //speed adjustment
            if (ref_vel > speed_limit)
            {
              ref_vel -= 0.224;              
            } else{
              ref_vel += 0.224;
            }
            //lane change
            if (lane_offset == center_lane)
            {
              if (left_cars.empty())
              {
                lane = 0;
              } else if (right_cars.empty()){
                lane = 2;
              }
            } else if (lane_offset == left_lane || lane_offset == right_lane){
              if (center_cars.empty())
              {
                lane = 1;
              }
            }
          } else if(ref_vel < speed_limit) {
            ref_vel += 0.224;
          }

          //creating list of widley spaced (x,y) waypoints, evenly spaced at 30m
          //later, we interpolate these points with a spline and fill with more points
          vector<double> ptsx;
          vector<double> ptsy;

          //reference x, y, yaw states
          // either reference the starting point as where the car is or at the previous paths end point
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);

          //if previous size is almost empy, use the car as starting reference
          if(prev_size < 2)
          {
            //use two points that make the path tanget to the car
            double prev_car_x = car_x - cos(car_yaw);
            double prev_car_y = car_y - sin(car_yaw);

            ptsx.push_back(prev_car_x);
            ptsx.push_back(car_x);

            ptsy.push_back(prev_car_y);
            ptsy.push_back(car_y);
          }
          //use previous path's end point as starting reference
          else
          {
            ref_x = previous_path_x[prev_size-1];
            ref_y = previous_path_y[prev_size-1];

            double ref_x_prev = previous_path_x[prev_size-2];
            double ref_y_prev = previous_path_y[prev_size-2];
            ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

            ptsx.push_back(ref_x_prev);
            ptsx.push_back(ref_x);

            ptsy.push_back(ref_y_prev);
            ptsy.push_back(ref_y);
          }

          //in Frenet, add evenly 30m spaced points ahead of the starting reference
          vector<double> next_wp0 = getXY(car_s+30, (lane_offset), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp1 = getXY(car_s+60, (lane_offset), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp2 = getXY(car_s+90, (lane_offset), map_waypoints_s, map_waypoints_x, map_waypoints_y);

          //add x,y points from wp0,wp1,wp2 to ptsx,ptsy vectors
          ptsx.push_back(next_wp0[0]);
          ptsx.push_back(next_wp1[0]);
          ptsx.push_back(next_wp2[0]);

          ptsy.push_back(next_wp0[1]);
          ptsy.push_back(next_wp1[1]);
          ptsy.push_back(next_wp2[1]);          

          //transform to local car coordinates
          for (int i = 0; i < ptsx.size(); ++i)
          {
            //shift car reference angle to 0 degrees
            double shift_x = ptsx[i]-ref_x;
            double shift_y = ptsy[i]-ref_y;

            ptsx[i] = (shift_x * cos(0-ref_yaw)-shift_y*sin(0-ref_yaw));
            ptsy[i] = (shift_x * sin(0-ref_yaw)+shift_y*cos(0-ref_yaw));
          }

          //create a spline
          tk::spline s;

          //set (x,y) points to the spline
          s.set_points(ptsx, ptsy);

          //define the actual (x,y) points we will use for the planner
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //start with all the previous path points from last iteration
          for (int i = 0; i < previous_path_x.size(); ++i)
          {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          //calculate how to break up spline points so we travel at desired velocity
          double target_x = 30.0;
          double target_y = s(target_x);
          double target_dist = sqrt((target_x)*(target_x)+(target_y)*(target_y));

          double x_add_on = 0;

          //fill up the rest of our path planner after filling it with previous points, here we will always output 50 points
          for (int i = 1; i <= 50-previous_path_x.size(); ++i)
          {
            double N = (target_dist/(0.02*ref_vel/2.24)); //ref_val/2.24 is to convert from mph to m/s
            double x_point = x_add_on+(target_x)/N;
            double y_point = s(x_point);

            x_add_on = x_point;

            double x_ref = x_point;
            double y_ref = y_point;

            //rotate back to normal after rotating it earlier
            x_point = (x_ref *cos(ref_yaw)-y_ref*sin(ref_yaw));
            y_point = (x_ref *sin(ref_yaw)+y_ref*cos(ref_yaw));

            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }

          /**
           * 
           * END OF MY CODE
           */

          json msgJson;
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}