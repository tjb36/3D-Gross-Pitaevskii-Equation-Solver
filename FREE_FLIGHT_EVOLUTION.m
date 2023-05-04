%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------ Free-Flight Evolution ---------------%
%------- T. Barrett, Uni. of Sussex. 2021 ---------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc;close all;
disp([datestr(now), ' : ', ' Starting ',mfilename,'.m']);
addpath(genpath('Tools'))
output_filename = "psi_tfinal.mat";

UseGPU = 1;      % Use GPU for the calculation if one is available

c = physical_constants();  % Structure to store required physical constants

%%% Input parameters in SI units %%%
N = 1e4;               % Number of particles in BEC
omega_x = 2*pi*20;     % x trap frequency in radians per second = 2*pi x Hz
omega_y = 2*pi*600;     % y trap frequency in radians per second = 2*pi x Hz
omega_z = 2*pi*600;       % z trap frequency in radians per second = 2*pi x Hz
t_final = 20e-3;          % Final expansion time to calculate wavefunction (ms)
t_start = 3e-3;          % Time that the GPE evolution was run until (ms) 6.3908e-3
t_flight = omega_x*(t_final - t_start); % Free flight time (dimensionless units)
l = sqrt( c.hbar/(c.mRb87*omega_x) );   % Unit of length ( radial harmonic oscillator size )

%%% Import and plot ground state %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('psi_ground.mat')
[~,x0_ind] = min(abs(x));  % Get indices of zero points for later slicing 
[~,y0_ind] = min(abs(y));  % ( should be (N/2 + 1) if N is even )
[~,z0_ind] = min(abs(z));  %
dx = x(2) - x(1);    % Original lattice spacings (dimensionless units)
dy = y(2) - y(1);    %
dz = z(2) - z(1);    %

f1 = figure('Position',[0.0842    0.1162    1.3368    0.6272]*1e3);
subplot(3,3,1); plot( squeeze(x) *l*1e6 , squeeze( abs(psi_k(y0_ind,:,z0_ind)).^2 ) /(l^3)*N/(1e18) ,'.-'); xlabel('x (\mum)'); ylabel('Density (\mum^-^3)')
subplot(3,3,2); plot( squeeze(y) *l*1e6 , squeeze( abs(psi_k(:,x0_ind,z0_ind)).^2 ) /(l^3)*N/(1e18) ,'.-'); xlabel('y (\mum)'); ylabel('Density (\mum^-^3)')
subplot(3,3,3); plot( squeeze(z) *l*1e6 , squeeze( abs(psi_k(y0_ind,x0_ind,:)).^2 ) /(l^3)*N/(1e18) ,'.-'); xlabel('z (\mum)'); ylabel('Density (\mum^-^3)')
annotation(f1,'textbox',...
    [0.00418910831837223 0.806397958545054 0.0993417137401109 0.0427295924753559],...
    'String',{'Ground State (t=0)'},...
    'LineStyle','none');
annotation(f1,'textbox',...
    [0.00867743865948532 0.48560714200565 0.073608619170871 0.0516581641168011],...
    'String',{'State at t_G_P_E'},...
    'LineStyle','none');
annotation(f1,'textbox',...
    [0.00748055056852183 0.192239795279748 0.0640335142148747 0.0427295924753559],...
    'String',{'Final State'},...
    'LineStyle','none');

x_ground = x;
y_ground = y;
z_ground = z;
n_1D_x_ground = sum( abs(psi_k).^2 / (l^3)*N , [1 3]) * dy*l * dz*l ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%% Import and plot t_GPE state %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('psi_tGPE.mat')

%%% Initial grid vectors %%%%%%%%%%%%%%%%%%%%%%%%%%
dx = x(2) - x(1);    % Original lattice spacings (dimensionless units)
dy = y(2) - y(1);    %
dz = z(2) - z(1);    %
Nx = size(psi_k,2);
Ny = size(psi_k,1);
Nz = size(psi_k,3);

%%%%%%%%%% Add initial extra zero-padding %%%%%%%%%%
% (to allow some initial room for expansion compared to starting state grid)
Nx_new = 64;
Ny_new = 192;
Nz_new = 192;
assert( all([Nx_new Ny_new Nz_new] >= [Nx Ny Nz]), 'Padded initial grid sizes cannot be smaller than those of the ground state')

[x, ~] = grid_vectors(Nx_new, dx); % Create new grid vectors
[y, ~] = grid_vectors(Ny_new, dy); %
[z, ~] = grid_vectors(Nz_new, dz); %
[~,x0_ind] = min(abs(x));  % Get indices of zero points for later slicing 
[~,y0_ind] = min(abs(y));  % ( should be (N/2 + 1) if N is even )
[~,z0_ind] = min(abs(z));  %

% Permute variables to enable use of Matlab's implicit expansion
x = permute(x, [1 2 3]); % Creates a [1 x Nx x 1] vector
y = permute(y, [2 1 3]); % Creates a [Ny x 1 x 1] vector
z = permute(z, [3 1 2]); % Creates a [1 x 1 x Nz] vector

% Pad the wavefunction with vacuum (zeros) to reach size of Nx_new x Ny_new x Nz_new
psi_k = padarray(psi_k, [Ny_new-Ny Nx_new-Nx Nz_new-Nz]/2, 0, 'both');

Nx = Nx_new;
Ny = Ny_new;
Nz = Nz_new;

%%% Imprint phase
% n_1D_x = sum( abs(psi_k).^2 / (l^3)*N , [1 3]) * dy*l * dz*l ;
% jmax = 20;
% Temperature = 230e-9;
% [x_phi,phi_j] = calculate_Petrov_phase(omega_x, omega_y, N, x*l, n_1D_x, Temperature, jmax);
% phi = sum(phi_j,1).';
% phi_padded = padarray(phi, (Nx-length(x_phi))/2, phi(end), 'post');
% phi_padded = padarray(phi_padded, (Nx-length(x_phi))/2, phi(1), 'pre');
% phi_matrix = zeros(size(psi_k));
% for jj = 1:Ny
%     for kk = 1:Nz
%         phi_matrix(jj,:,kk) = phi_padded;
%     end
% end
% psi_k = psi_k .* exp(1i*phi_matrix);

% phi = 0.4*cos(2*pi*0.3*x + 0);
% psi_k = psi_k.*exp(1i*phi);

%%%


figure(f1)
subplot(3,3,4); plot( squeeze(x) *l*1e6 , squeeze( abs(psi_k(y0_ind,:,z0_ind)).^2 ) /(l^3)*N/(1e18) ,'.-'); xlabel('x (\mum)'); ylabel('Density (\mum^-^3)')
subplot(3,3,5); plot( squeeze(y) *l*1e6 , squeeze( abs(psi_k(:,x0_ind,z0_ind)).^2 ) /(l^3)*N/(1e18) ,'.-'); xlabel('y (\mum)'); ylabel('Density (\mum^-^3)')
subplot(3,3,6); plot( squeeze(z) *l*1e6 , squeeze( abs(psi_k(y0_ind,x0_ind,:)).^2 ) /(l^3)*N/(1e18) ,'.-'); xlabel('z (\mum)'); ylabel('Density (\mum^-^3)')
drawnow()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Perform the free-flight
multx = 1; % Lattice expansion multipliers
multy = 5; %
multz = 5; %

[x_final, y_final, z_final, psi_final] = free_flight(psi_k, dx, dy, dz, multx, multy, multz, t_flight, UseGPU);

dxf = x_final(2) - x_final(1);
dyf = y_final(2) - y_final(1);
dzf = z_final(2) - z_final(1);
INT_final = sum(sum(sum( abs(psi_final).^2 /(l^3)*N ) ) )*dxf*dyf*dzf*l^3;
n_1D_x_tfinal = sum( abs(psi_final).^2 / (l^3)*N , [1 3]) * dyf*l * dzf*l ; % final integrated density
n2D_xz_int = squeeze( sum( abs(psi_final).^2 / (l^3)*N ,1) * dyf*l ) .';

%%% Save final state %%%%%%%%%%%%%%%%%
save('outputs\' + output_filename,'psi_final','x_final','y_final','z_final')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
f2 = figure;
isoval = max(abs(psi_final(:)).^2)/100;
s = isosurface(x_final*l*1e6 , y_final*l*1e6 , z_final*l*1e6, abs(psi_final).^2 , isoval);
p = patch(s);
isonormals(x_final*l*1e6 , y_final*l*1e6 , z_final*l*1e6, abs(psi_final).^2 , p)
set(gca,'FontSize',14)
xlabel('$x\,(\mu\textrm{m})$','Interpreter','Latex');
ylabel('$y\,(\mu\textrm{m})$','Interpreter','Latex');
zlabel('$z\,(\mu\textrm{m})$','Interpreter','Latex');
title('Isosurface of 1% Density Level','FontSize',14)
view(3);
set(p,'EdgeColor','none');
p.SpecularStrength = 0.4;
camlight;
camlight('headlight');
camlight('left');
lighting gouraud;
set(p,'FaceColor',[33,113,181]/255);
grid on;box on;
axis equal
alpha(0.9)

figure;
imagesc(squeeze(x_final)*l*1e6,squeeze(z_final)*l*1e6,n2D_xz_int);axis equal tight;
xlabel('x (\mum)');ylabel('z (\mum)');
colormap(viridis(100))
colorbar
set(gca, 'YDir', 'normal')

figure(f1)
subplot(3,3,7); plot( squeeze(x_final) *l*1e6 , squeeze( abs(psi_final(y0_ind,:,z0_ind)).^2 ) /(l^3)*N/(1e18) ,'.-'); xlabel('x (\mum)'); ylabel('Density (\mum^-^3)')
subplot(3,3,8); plot( squeeze(y_final) *l*1e6 , squeeze( abs(psi_final(:,x0_ind,z0_ind)).^2 ) /(l^3)*N/(1e18) ,'.-'); xlabel('y (\mum)'); ylabel('Density (\mum^-^3)')
subplot(3,3,9); plot( squeeze(z_final) *l*1e6 , squeeze( abs(psi_final(y0_ind,x0_ind,:)).^2 ) /(l^3)*N/(1e18) ,'.-'); xlabel('z (\mum)'); ylabel('Density (\mum^-^3)')

%%
%%% Castin-Dum Result after full TOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_final_fine = linspace(x_final(1),x_final(end),500);
y_final_fine = linspace(y_final(1),y_final(end),500);
z_final_fine = linspace(z_final(1),z_final(end),500);

[R_x_t, R_y_t, R_z_t] = CastinDumRadii(omega_x, omega_y, omega_z, N, t_final);

n_1D_CastinDum = 15*N/(16*R_x_t) * max( 1-(x_final_fine*l/R_x_t).^2  ,0  ).^2  ;
n_x_CastinDum =  15/(8*pi)*N/(R_x_t*R_y_t*R_z_t) * max( (1 - (x_final_fine*l).^2/R_x_t^2 )  , 0 );
n_y_CastinDum =  15/(8*pi)*N/(R_x_t*R_y_t*R_z_t) * max( (1 - (y_final_fine*l).^2/R_y_t^2 )  , 0 );
n_z_CastinDum =  15/(8*pi)*N/(R_x_t*R_y_t*R_z_t) * max( (1 - (z_final_fine*l).^2/R_z_t^2 )  , 0 );
subplot(3,3,7); hold on; plot( x_final_fine*l*1e6 , n_x_CastinDum /1e18 )
subplot(3,3,8); hold on; plot( y_final_fine*l*1e6 , n_y_CastinDum /1e18 )
subplot(3,3,9); hold on; plot( z_final_fine*l*1e6 , n_z_CastinDum /1e18 )
legend('Numerical','Castin-Dum')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Thomas-Fermi Profile In-Trap %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_ground_fine = linspace(x_ground(1),x_ground(end),500);
y_ground_fine = linspace(y_ground(1),y_ground(end),500);
z_ground_fine = linspace(z_ground(1),z_ground(end),500);
mu3D = ( 15*sqrt(2)/(32*pi) * N * c.g_int3D *c.mRb87^(3/2) * omega_x*omega_y*omega_z )^(2/5); % chemical potential (Thomas-Fermi limit)
R_x_0  = sqrt( 2*mu3D/(c.mRb87*omega_x^2) );   % Thomas-Fermi radius in-situ
R_y_0  = sqrt( 2*mu3D/(c.mRb87*omega_y^2) );   
R_z_0  = sqrt( 2*mu3D/(c.mRb87*omega_z^2) );   
n_1D_x_TF = 15*N/(16*R_x_0) * max( 1-(x_ground_fine*l/R_x_0).^2  ,0  ).^2  ; % in-trap Thomas-Fermi density
n_x_TF =  15/(8*pi)*N/(R_x_0*R_y_0*R_z_0) * max( (1 - (x_ground_fine*l).^2/R_x_0^2 )  , 0 );
n_y_TF =  15/(8*pi)*N/(R_x_0*R_y_0*R_z_0) * max( (1 - (y_ground_fine*l).^2/R_y_0^2 )  , 0 );
n_z_TF =  15/(8*pi)*N/(R_x_0*R_y_0*R_z_0) * max( (1 - (z_ground_fine*l).^2/R_z_0^2 )  , 0 );
subplot(3,3,1); hold on; plot( x_ground_fine*l*1e6 , n_x_TF /1e18 )
subplot(3,3,2); hold on; plot( y_ground_fine*l*1e6 , n_y_TF /1e18 )
subplot(3,3,3); hold on; plot( z_ground_fine*l*1e6 , n_z_TF /1e18 )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Plot integrated 1D densities %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;hold all;
plot(x_ground_fine*l*1e6, n_1D_x_TF)
plot(x_ground*l*1e6, n_1D_x_ground)
plot(x_final_fine*l*1e6, n_1D_CastinDum)
plot(x_final*l*1e6, n_1D_x_tfinal)
title(sprintf('1D Density after %.1f ms Free Flight',t_flight/omega_x*1e3))
ylabel('n_1_d(x) [atoms/m] ');
xlabel('x [\mum]')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

legend('In-Trap Thomas-Fermi Profile','Numerical Ground State','Castin-Dum Theory','Final Numerical 1D density')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Free Flight Prescriptor %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (according to E. 42 of Deuar,P., Comp. Phys. Comm. 208, 92-102 (2016) %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xf, yf, zf, psi_out] = free_flight(psi_in, ...
    dxo, dyo, dzo, mult_x, mult_y, mult_z, t_flight, UseGPU)

% suffix = o : denotes "original" lattice variables
% suffix = f : denotes "final" lattice variables

tic
fprintf('Running free flight evolution... \n');

Nx = size(psi_in,2);
Ny = size(psi_in,1);
Nz = size(psi_in,3);
M = Nx * Ny * Nz;  % total number of grid points

Nx_32 = uint32( Nx );
Ny_32 = uint32( Ny );
Nz_32 = uint32( Nz );
M_32 = uint32( M );
IND = uint32( 0:(M_32-1) );

num_mu = mult_x * mult_y * mult_z;   % number of terms in sum

dxf = dxo * mult_x;   % Final lattice real space grid spacings
dyf = dyo * mult_y;
dzf = dzo * mult_z;

Lxo = dxo * Nx;       % Original lattice box lengths
Lyo = dyo * Ny;
Lzo = dzo * Nz;

Lxf = Lxo * mult_x;   % Final lattice box lengths
Lyf = Lyo * mult_y;
Lzf = Lzo * mult_z;

dkxo = 2*pi / Lxo;    % Original lattice k space grid spacings
dkyo = 2*pi / Lyo;
dkzo = 2*pi / Lzo;

dkxf = dkxo / mult_x; % Final lattice k space grid spacings
dkyf = dkyo / mult_y; 
dkzf = dkzo / mult_z;

axo = -Lxo / 2;       % Original lattice offsets
ayo = -Lyo / 2;
azo = -Lzo / 2;

axf = - Lxf / 2;      % Final lattice offsets
ayf = - Lyf / 2;
azf = - Lzf / 2;

xo = ( 0:dxo:(Nx-1)*dxo ) + axo; % Original lattice real space grid vectors
yo = ( 0:dyo:(Ny-1)*dyo ) + ayo;
zo = ( 0:dzo:(Nz-1)*dzo ) + azo;

% Change dimension orientation to allow implicit expansion in Matlab
xo = permute(xo, [1 2 3]); % Creates a [1 x Nx x 1] vector
yo = permute(yo, [2 1 3]); % Creates a [Ny x 1 x 1] vector
zo = permute(zo, [3 1 2]); % Creates a [1 x 1 x Nz] vector

xf = (xo * mult_x); % Final lattice real space grid vectors
yf = (yo * mult_y);
zf = (zo * mult_z);

% Final lattice k space grid vectors
kxf = ifftshift( (-pi/dxf):dkxf:(pi/dxf - dkxf) );
kyf = ifftshift( (-pi/dyf):dkyf:(pi/dyf - dkyf) );
kzf = ifftshift( (-pi/dzf):dkzf:(pi/dzf - dkzf) );

kxo = kxf * mult_x; % Original lattice k space grid vectors
kyo = kyf * mult_y;
kzo = kzf * mult_z;

% Change dimension orientation to allow implicit expansion in Matlab
kxo = permute(kxo, [1 2 3]); % Creates a [1 x Nx x 1] vector
kyo = permute(kyo, [2 1 3]); % Creates a [Ny x 1 x 1] vector
kzo = permute(kzo, [3 1 2]); % Creates a [1 x 1 x Nz] vector

%%% PREFACTORS WHICH DO NOT DEPEND ON LOOP VARIABLES %%%          
Ekfacx = exp( -1i/2 * t_flight * kxo.^2 );  % Prefactors used in Eq. (42c) 
Ekfacy = exp( -1i/2 * t_flight * kyo.^2 );
Ekfacz = exp( -1i/2 * t_flight * kzo.^2 );
kdafacx = exp( 1i * kxo * (axf-axo) );       % Prefactors used in Eq. (42c) 
kdafacy = exp( 1i * kyo * (ayf-ayo) );
kdafacz = exp( 1i * kzo * (azf-azo) );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

psi_out = zeros(Ny,Nx,Nz); % Preallocate final wavefunction array

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------ Move arrays to GPU memory -----------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if UseGPU
    try
        gpuArray(1);
        UseGPU = 1;
        fprintf('GPU device available to use.\n')
        
        psi_in = gpuArray(psi_in);
        psi_out = gpuArray(psi_out);
        xo = gpuArray(xo);
        yo = gpuArray(yo);
        zo = gpuArray(zo);
        kxo = gpuArray(kxo);
        kyo = gpuArray(kyo);
        kzo = gpuArray(kzo);
        xf = gpuArray(xf);
        yf = gpuArray(yf);
        zf = gpuArray(zf);
    catch ME
        UseGPU = 0;
        fprintf('GPU device not available to use.\n')
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for mu_x = 0:(mult_x-1)
    for mu_y = 0:(mult_y-1)
        for mu_z = 0:(mult_z-1)
            
            %%% PREFACTORS WHICH DEPEND ON LOOP VARIABLES %%%          
            xofac = exp( -1i*xo*mu_x*dkxo/mult_x ); % Prefactors used in exponential in Eq. (42d) 
            yofac = exp( -1i*yo*mu_y*dkyo/mult_y );
            zofac = exp( -1i*zo*mu_z*dkzo/mult_z );
            
            kxofac = exp( -1i*kxo*t_flight*mu_x*dkxo/mult_x ); % Prefactors used in exponential in Eq. (42c) 
            kyofac = exp( -1i*kyo*t_flight*mu_y*dkyo/mult_y );
            kzofac = exp( -1i*kzo*t_flight*mu_z*dkzo/mult_z );
            
            xffac  = exp( 1i*xf*mu_x*dkxo/mult_x ); % Prefactors used in exponential in Eq. (42b) 
            yffac  = exp( 1i*yf*mu_y*dkyo/mult_y );
            zffac  = exp( 1i*zf*mu_z*dkzo/mult_z );
            
            % Prefactor used in exponential in Eq. (42b) 
            dtfac  = exp( -1i/2 * t_flight *( (mu_x*dkxo/mult_x)^2 + (mu_y*dkyo/mult_y)^2 + (mu_z*dkzo/mult_z)^2 ) );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            fieldmu = psi_in .* xofac .* yofac .* zofac; % produces kernel for A in Eq.42d
            fieldmu = fftn(fieldmu);	                    % produces A in Eq.42d
            fieldmu = fieldmu .* (Ekfacx.*Ekfacy.*Ekfacz) .* ...
                                 (kdafacx.*kdafacy.*kdafacz) .* ...
                                 (kxofac.*kyofac.*kzofac);  % kernel for B in Eq.42c
            fieldmu = ifftn(fieldmu);	                    % produces B in Eq.42c
            
            % Final part calculates f in Eq. 42b and accumulates to sum in Eq. 42a
            J1 = idivide( idivide(IND,Nz_32), Ny_32 ); % idivide to avoid "double" casting, does "fix" rounding
            J2 = mod( idivide( IND, Nz_32 ), Ny_32 ); 
            J3 = mod( IND, Nz_32 );
            J1 = mod( mult_x*J1, Nx_32 );
            J2 = mod( mult_y*J2, Ny_32 );
            J3 = mod( mult_z*J3, Nz_32 );
            ind_prime = mod( ( J3 + Nz_32*(J2 + Ny_32*J1) ) , M_32 ) + 1;
            fieldmu = reshape( permute(fieldmu,[3,1,2]) , [M 1] ); % Convert 3D form back to 1D form, ready to use ind_prime
            psi_out = psi_out +  permute( reshape( fieldmu(ind_prime), ...
                [Nz,Ny,Nx] ) , [2,3,1] ) .* (xffac .* yffac .* zffac) * dtfac;

            fprintf("Accumulated [%i %i %i] multiplier (%i of %li) ... (%.0f%% complete) \n",...
                mu_x,mu_y,mu_z, mu_z + mu_y*mult_z + mu_x*(mult_y*mult_z) +1 , num_mu, ...
                (mu_z + mu_y*mult_z + mu_x*(mult_y*mult_z) +1)/num_mu*100);
        end
    end
end

psi_out = psi_out / num_mu;  % Correct normalisation after lattice sums

% Bring results back from GPU, if using
if UseGPU
    psi_out = gather(psi_out);
    xf = gather(xf);
    yf = gather(yf);
    zf = gather(zf);
end

toc
end


function [x, kx] = grid_vectors(Nx, dx)
%%% Function for generating real space and frequency vectors in each dirn.

Fsx = 1/dx;         % Sampling frequency
dfx = Fsx/Nx;       % Frequency resolution

Nx_IS_EVEN =  ~rem(Nx,2);
if Nx_IS_EVEN
    x  = -(Nx*dx/2):dx:(Nx*dx/2-dx); % Real space vector
    fx = -(Fsx/2):dfx:(Fsx/2-dfx);   % Spatial frequency vector
else
    x  = -((Nx-1)*dx/2):dx:((Nx-1)*dx/2);
    fx = -(Fsx/2-dfx/2):dfx:(Fsx/2-dfx/2);
end

kx = 2*pi*fx;       % Angular spatial frequency k vector

end