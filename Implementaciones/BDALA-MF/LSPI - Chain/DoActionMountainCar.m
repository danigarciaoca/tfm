function [xp, reward, terminal] = DoActionMountainCar( a, x, env )
%DoActionMountainCar: executes the action (a) into the mountain car
%environment
% a: is the force to be applied to the car
% x: is the vector containning the position and speed of the car
% xp: is the vector containing the new position and speed of the car


% Get environment variables
bound_speed_left = env.bound_speed_left; 
bound_speed_right = env.bound_speed_right;
bound_position_left = env.bound_position_left;
goal = env.goal;
friction = env.friction;
time_step = env.time_step;
slope = env.slope;
slope_ampli = env.slope_ampli;
actions_list = env.actions_list;


% Get current state
position = x(1);
speed    = x(2); 

% Get current action
force = actions_list(a);

% speed state-component transition 
speed_t1 = speed + (time_step*force) + (slope * cos( slope_ampli*position) );	 
% speed_t1 = speed_t1 * friction; % include friction for a more realistic simulation.

if(speed_t1 < bound_speed_left) 
    speed_t1 = bound_speed_left; 
end
if(speed_t1 > bound_speed_right)
    speed_t1 = bound_speed_right; 
end

% Position state-component transition 
position_t1 = position + speed_t1; 

if(position_t1 < bound_position_left)
    position_t1 = bound_position_left;
    speed_t1 = 0.0;
end

% Aggregate state
xp = [position_t1, speed_t1];

% Get reward and identify terminal state
if(position_t1 >= goal)
    reward = 0;
    terminal = true;
else
    reward = -1;
    terminal = false;
end


end % function DoActionMountainCar
