function [R_x_t, R_y_t, R_z_t] = CastinDumRadii(omega_x_0, omega_y_0, omega_z_0, N, tmax)

c = physical_constants();  % Structure to store required physical constants

dt = 0.1e-3;    % Time step
t = 0:dt:tmax;  % Create time vector

% New trap frequencies
omega_x_t = 2*pi*0;
omega_y_t = 2*pi*0;
omega_z_t = 2*pi*0;

% Initial conditions
lambda_x_0 = 1;
lambda_y_0 = 1;
lambda_z_0 = 1;
dlambda_x_dt_0 = 0;
dlambda_y_dt_0 = 0;
dlambda_z_dt_0 = 0;
initial_conds = [lambda_x_0 dlambda_x_dt_0 ...
                 lambda_y_0 dlambda_y_dt_0 ...
                 lambda_z_0 dlambda_z_dt_0];

% Solve the Castin-Dum scaling equations 
[t, x] = ode45( @(t,x) rhs(t, x, omega_x_0, omega_y_0, omega_z_0, ...
                                 omega_x_t, omega_y_t, omega_z_t), t, initial_conds );
                             
mu_3D = (15*sqrt(2)/(32*pi)*N*c.g_int3D*c.mRb87^(3/2)*omega_x_0*omega_y_0*omega_z_0)^(2/5);
R_x_0 = sqrt( 2*mu_3D/(c.mRb87*omega_x_0^2) );
R_y_0 = sqrt( 2*mu_3D/(c.mRb87*omega_y_0^2) );
R_z_0 = sqrt( 2*mu_3D/(c.mRb87*omega_z_0^2) );

R_x_t = x(end,1)*R_x_0;
R_y_t = x(end,3)*R_y_0;
R_z_t = x(end,5)*R_z_0;

end

function Xd = rhs(t, x, omega_x_0, omega_y_0, omega_z_0, omega_x_t, omega_y_t, omega_z_t)

lambda_x_t     = x(1);
dlambda_x_dt_t = x(2);
lambda_y_t     = x(3);
dlambda_y_dt_t = x(4);
lambda_z_t     = x(5);
dlambda_z_dt_t = x(6);

Xd = [ dlambda_x_dt_t;...
       omega_x_0^2/(lambda_x_t^2*lambda_y_t*lambda_z_t) - omega_x_t^2*lambda_x_t;...
       dlambda_y_dt_t;...
       omega_y_0^2/(lambda_x_t*lambda_y_t^2*lambda_z_t) - omega_y_t^2*lambda_y_t;...
       dlambda_z_dt_t;...
       omega_z_0^2/(lambda_x_t*lambda_y_t*lambda_z_t^2) - omega_z_t^2*lambda_z_t;];

end