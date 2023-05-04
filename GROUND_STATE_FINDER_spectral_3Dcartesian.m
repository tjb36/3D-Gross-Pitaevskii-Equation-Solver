%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------- Imaginary Time GPE Evolution ------------%
%--------- T. Barrett, Uni. of Sussex. 2021 ----------%
%-----------------------------------------------------%
% Calculates the ground state of the GPE for a BEC in %
% a given trap using the imaginary time method.       %
% The equation is evolved with the Split-Step Fourier %
% technique using second order Strang Splitting, in   %
% 3D cartesian coordinates.                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc;close all;
disp([datestr(now), ' : ', ' Starting ',mfilename,'.m']);
addpath(genpath('tools'))
c = physical_constants(); % Structure to store required physical constants

output_filename = "psi_ground.mat";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------- Input parameters -----------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UseGPU = 0;      % Use GPU for the calculation if one is available
updatePlots = 1; % Update plots periodically throughout simulation
storeValues = 1; % Store values (energy, mu, wavefunction cuts, etc)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------- Trap parameters -----------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 1e4;                % Number of particles in BEC
omega_x = 2*pi*20;      % x trap frequency in radians per second = 2*pi x Hz
omega_y = 2*pi*600;     % y trap frequency in radians per second = 2*pi x Hz
omega_z = 2*pi*600;     % z trap frequency in radians per second = 2*pi x Hz

%%% Dimensionless parameters %%%%%%%%%%%%%%%%%%%%%%%
kappa = omega_y/omega_x;                % Anisotropy parameter 1
lambda = omega_z/omega_x;               % Anisotropy parameter 2
l = sqrt( c.hbar/(c.mRb87*omega_x) );   % Unit of length ( x-dir harmonic oscillator size )
G = 4*pi*c.aRb87*N/l;                   % Scaled three-dimensional non-linearity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------- Imaginary Time Grid ---------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time variables are dimensionless. t, dt, and T here are scaled
% from their real SI values (which would be in seconds) by omega_x 
% (e.g. t = omega_x * t_SI)
dt =  0.00001;      % Imaginary time step (dimensionless)
Nt = 10000;         % Number of time iterations
T = (Nt-1)*dt;      % Duration of total time window (dimensionless)
t = 0:dt:T;         % Define imaginary time vector (dimensionless)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------ Spatial Grids -----------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Spatial variables are dimensionless. dx, dy, and dz here are scaled
% from their real SI values (which would be in metres) by l, which is defined
% above to be the harmonic oscillator length in the x-direction.
% ( e.g. dx = dx_SI / l ). For omega_x = 2pi x 1kHz, l = 0.34 um
Nx = 64;         % Number of x grid points ( make this power of 2 for FFT speed )
dx = 0.8;        % Spacing of x grid
Ny = 16;         % Number of y grid points ( make this power of 2 for FFT speed )
dy = 0.13;       % Spacing of y grid
Nz = 16;         % Number of y grid points ( make this power of 2 for FFT speed )
dz = 0.13;       % Spacing of y grid

dV = dx*dy*dz;   % Volume element

[x, kx] = grid_vectors(Nx, dx);
[y, ky] = grid_vectors(Ny, dy);
[z, kz] = grid_vectors(Nz, dz);

% Permute variables to enable use of Matlab's implicit expansion
x = permute(x, [1 2 3]); % Creates a [1 x Nx x 1] vector
y = permute(y, [2 1 3]); % Creates a [Ny x 1 x 1] vector
z = permute(z, [3 1 2]); % Creates a [1 x 1 x Nz] vector

[~,x0_ind] = min(abs(x));  % Get indices of zero points for later slicing 
[~,y0_ind] = min(abs(y));  % ( should be (N/2 + 1) if N is even )
[~,z0_ind] = min(abs(z));  %        

kx = permute(ifftshift(kx), [1 2 3]);
ky = permute(ifftshift(ky), [2 1 3]);
kz = permute(ifftshift(kz), [3 1 2]);

Tx = exp( -dt/2 * kx.^2); % Kinetic energy operators
Ty = exp( -dt/2 * ky.^2);
Tz = exp( -dt/2 * kz.^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------  External trapping potential -----------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (in dimensionaless energy units of hbar*omega_x)
V_ext = (1/2)*(x.^2 + kappa^2*y.^2 + lambda^2*z.^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Calculate some analytic limits for the density profiles %%%
n_ThomasFermi_3D = ThomasFermi_3D_density(x*l, y*l, z*l, omega_x, omega_y, omega_z, N, c);
n_NonInteracting_3D = NonInteracting_3D_density(x*l, y*l, z*l, omega_x, omega_y, omega_z, N, c);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------- Initial Condition ----------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a_ho = sqrt(c.hbar/(c.mRb87*(omega_x*omega_y*omega_z)^(1/3)));
ThomasFermiParameter = N*c.aRb87/a_ho; % Measures significance of interactions
if ThomasFermiParameter > 3 % If interactions are strong, use ThomasFermi profile
    psi_k = sqrt( n_ThomasFermi_3D / N ) * sqrt(l^3);
else                        % If interactions are weak, use non-interacting ground state
    psi_k = sqrt( n_NonInteracting_3D / N ) * sqrt(l^3);
end

INT_psi_initial = sum(sum(sum( abs(psi_k).^2 )))*dV; % Integrate initial distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------ Set Up Plots ------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if updatePlots
    figure('Position',1e3*[0.0674    0.3258    1.3976    0.2880],'color','w');
    h.ax1 = subplot(1,3,1);hold all;
    h.p1 = plot( squeeze(x)*l*1e6, squeeze( abs(psi_k(y0_ind,:,z0_ind)).^2 ) /(l^3)*N/1e18 ,'.-');
    plot( squeeze(x)*l*1e6, squeeze( n_ThomasFermi_3D(y0_ind,:,z0_ind) )/1e18 );
    plot( squeeze(x)*l*1e6, squeeze( n_NonInteracting_3D(y0_ind,:,z0_ind) ) /1e18 );
    xlabel('$x\,(\mu\textrm{m})$','Interpreter','Latex');ylabel('$n(x,0,0)\,(\mu\textrm{m}^{-3})$','Interpreter','Latex');box on;title('X Cut')
    legend('Numerical result','Thomas-Fermi','Non-interacting','location','SouthWest')

    h.ax2 = subplot(1,3,2);hold all;
    h.p2 = plot( squeeze(y)*l*1e6, squeeze( abs(psi_k(:,x0_ind,z0_ind)).^2 ) /(l^3)*N/1e18 ,'.-');
    plot( squeeze(y)*l*1e6, squeeze( n_ThomasFermi_3D(:,x0_ind,z0_ind) /1e18 ) );
    plot( squeeze(y)*l*1e6, squeeze( n_NonInteracting_3D(:,x0_ind,z0_ind) /1e18 ) );
    xlabel('$y\,(\mu\textrm{m})$','Interpreter','Latex');ylabel('$n(0,y,0)\,(\mu\textrm{m}^{-3})$','Interpreter','Latex');box on;title('Y Cut')
    
    h.ax3 = subplot(1,3,3);hold all;
    h.p3 = plot( squeeze(z)*l*1e6, squeeze( abs(psi_k(y0_ind,x0_ind,:)).^2 ) /(l^3)*N/1e18 ,'.-');
    plot( squeeze(z)*l*1e6, squeeze( n_ThomasFermi_3D(y0_ind,x0_ind,:) /1e18 ) );
    plot( squeeze(z)*l*1e6, squeeze( n_NonInteracting_3D(y0_ind,x0_ind,:) /1e18 ) );
    xlabel('$z\,(\mu\textrm{m})$','Interpreter','Latex');ylabel('$n(0,0,z)\,(\mu\textrm{m}^{-3})$','Interpreter','Latex');box on;title('Z Cut')
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------- Set up storage vectors/matrices ---------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dStore = 500;    % Only store every 'dStore' iterations
if storeValues
    num_to_store = ceil(Nt/dStore);              % Total number of iterations that will be stored
    t_store = zeros( 1, num_to_store );          % Preallocate storage arrays
    Etot_store = zeros( 7, num_to_store );       % Preallocate storage arrays
    mu_store = zeros( 1, num_to_store );         % Preallocate storage arrays
    density_store.xcut = zeros(Nx,num_to_store);
    density_store.ycut = zeros(Ny,num_to_store);
    density_store.zcut = zeros(Nz,num_to_store);
    
    %%% Store the first value in each storage matrix %%%
    store_ind = 1;
    t_store(store_ind) = t(1);
    [Etot_store(:,store_ind), mu_store(store_ind)] = calculate_energy_and_mu(psi_k, dV, kx, ky, kz, V_ext, G);
    density_store.xcut(1:Nx,store_ind) = squeeze( abs(psi_k(y0_ind,:,z0_ind)).^2 );
    density_store.ycut(1:Ny,store_ind) = squeeze( abs(psi_k(:,x0_ind,z0_ind)).^2 );
    density_store.zcut(1:Nz,store_ind) = squeeze( abs(psi_k(y0_ind,x0_ind,:)).^2 );
    
    disp('--------------')
    fprintf('Storing wavefunction, iteration 1 of %i: Energy = %0.10f, mu = %0.10f \n', Nt, Etot_store(7,store_ind), mu_store(store_ind));

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------ Move arrays to GPU memory -----------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(UseGPU)
    try
        gpuArray(1);
        UseGPU = 1;
        fprintf('GPU device available to use...\n')
    catch ME
        UseGPU = 0;
        fprintf('GPU device not available to use...\n')
    end
end

if UseGPU
    psi_k = gpuArray(psi_k); % Iterated wavefunction
    V_ext = gpuArray(V_ext); % External potential matrix
    Tx = gpuArray(Tx);       % Kinetic energy operator in x
    Ty = gpuArray(Ty);       % Kinetic energy operator in y
    Tz = gpuArray(Tz);       % Kinetic energy operator in z
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------ Main Time Stepping Loop -------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pause(1)
tic
fprintf('Running imaginary time evolution... \n');
for k = 2:Nt
    
    %%% Split-Step Spectral Evolution (using Strang Splitting) %%%
    psi_k = exp( -dt/2*(  V_ext + G*abs(psi_k).^2  ) ).*psi_k;         % Half step with potential operator
    psi_k = fftshift( ifftn( Tx.*Ty.*Tz.*fftn( ifftshift(psi_k) ) ) ); % Full step with kinetic operator (using implicit expansion)
    psi_k = exp( -dt/2*(  V_ext + G*abs(psi_k).^2  ) ).*psi_k;         % Half step with potential operator
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Renormalise the wavefunction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    INT_psi = sum(sum(sum( abs(psi_k).^2 )))*dV; % Calculate the integral
    psi_k = psi_k / sqrt(INT_psi);               % Renormalise
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%% Store Values Every dStore Iterations %%%%%%%%%%%%%%%%%%%
    if (mod(k,dStore) == 1 || dStore == 1) && storeValues
        
        store_ind = store_ind + 1;
        t_store(store_ind) = t(k);
        
        [E, mu] = calculate_energy_and_mu(psi_k, dV, kx, ky, kz, V_ext, G);
        fprintf('Storing wavefunction, iteration %i of %i: Energy = %0.10f, mu = %0.10f \n', k, Nt, E(7),mu);
        Etot_store(:,store_ind) = E;
        mu_store(:,store_ind) = mu;
        
        if UseGPU
            psi_k = gather(psi_k); % Bring current wavefunction back from GPU memory            
        end
        
        % Store the wavefunction line cuts
        density_store.xcut(1:Nx,store_ind) = squeeze( abs(psi_k(y0_ind,:,z0_ind)).^2 ); % Store the x line cut
        density_store.ycut(1:Ny,store_ind) = squeeze( abs(psi_k(:,x0_ind,z0_ind)).^2 ); % Store the y line cut
        density_store.zcut(1:Nz,store_ind) = squeeze( abs(psi_k(y0_ind,x0_ind,:)).^2 ); % Store the z line cut
        
        if UseGPU
            psi_k = gpuArray(psi_k); % Take current wavefunction back to GPU memory
        end
        
    end
    
    %%% Display percentage completed %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (mod(k,floor(Nt/10)) == 0) && updatePlots
        fprintf('Iteration %i of %i (%.0f%% complete)\n', k, Nt, k/Nt*100);
        
        if UseGPU
            psi_k = gather(psi_k); % Bring current wavefunction back from GPU memory
        end
        
        % Update plots
        set(h.p1,'YData', squeeze( abs(psi_k(y0_ind,:,z0_ind)).^2 ) /(l^3)*N/1e18 );
        set(h.p2,'YData', squeeze( abs(psi_k(:,x0_ind,z0_ind)).^2 ) /(l^3)*N/1e18  );
        set(h.p3,'YData', squeeze( abs(psi_k(y0_ind,x0_ind,:)).^2 ) /(l^3)*N/1e18 );
        set(h.ax1, 'YLim', [-0.01 1.1]*max(get(h.p1,'YData') ) )
        set(h.ax2, 'YLim', [-0.01 1.1]*max(get(h.p2,'YData') ) )
        set(h.ax3, 'YLim', [-0.01 1.1]*max(get(h.p3,'YData') ) )
        drawnow
        
        if UseGPU
            psi_k = gpuArray(psi_k); % Take current wavefunction back to GPU memory
        end
        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end


%%% Values from the final iteration %%%
[E, mu] = calculate_energy_and_mu(psi_k, dV, kx, ky, kz, V_ext, G);
E_final = E;
mu_final = mu;

fprintf('\n');
if UseGPU
    psi_k = gather(psi_k);
end
toc

mu_3D = (15*sqrt(2)/(32*pi)*N*c.g_int3D*c.mRb87^(3/2)*omega_x*omega_y*omega_z)^(2/5) / (c.hbar*omega_x);

fprintf('\n');
fprintf('=============================================================\n');
fprintf('----- Final Ground State Energy Components per Particle -----\n');
fprintf('-------------------- (in units of ℏ*ωx) ---------------------\n');
fprintf('=============================================================\n');
fprintf('Kinetic (x-direction)  = %0.6f \n', E_final(1,1));
fprintf('Kinetic (y-direction)  = %0.6f \n', E_final(2,1));
fprintf('Kinetic (z-direction)  = %0.6f \n', E_final(3,1));
fprintf('Kinetic (total)        = %0.6f \n', E_final(4,1));
fprintf('Potential              = %0.6f \n', E_final(5,1));
fprintf('Interaction            = %0.6f \n', E_final(6,1));
fprintf('Total                  = %0.6f \n', E_final(7,1));
fprintf('Chemical Potential     = %0.6f \n', mu_final);
fprintf('\n');
fprintf('Expected chemical potential in Thomas-Fermi limit = %0.2f \n', mu_3D);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Save final state %%%%%%%%%%%%%%%%%
save('outputs\' + output_filename,'psi_k','x','y','z')

% omega_ho = (omega_x*omega_y*omega_z)^(1/3);
% Tc = 0.94*c.hbar/c.kB*omega_ho*N^(1/3); % BEC transition temperature in harmonic trap
% mu_SI = mu_final * c.hbar * omega_x;
% mu_SI / (3/2*c.hbar*omega_x);
% mu_SI / (3/2*c.hbar*omega_y);
% mu_SI / (3/2*c.hbar*omega_z);
%%%


%% Plot 3D Isosurface
figure('Position',[64   62  560  420],'color','w');
isoval = max(abs(psi_k(:)).^2)/100;
s = isosurface(x*l*1e6, y*l*1e6, z*l*1e6, abs(psi_k).^2,isoval);
p = patch(s);
isonormals(x*l*1e6,y*l*1e6,z*l*1e6,abs(psi_k).^2,p)
set(gca,'FontSize',14)
xlabel('$x\,(\mu\textrm{m})$','Interpreter','Latex');
ylabel('$y\,(\mu\textrm{m})$','Interpreter','Latex');
zlabel('$z\,(\mu\textrm{m})$','Interpreter','Latex');
title('Isosurface of 1% Density Level','FontSize',14)
view(3);
set(p,'EdgeColor','none');
set(p,'FaceColor',[33,113,181]/255);
p.SpecularStrength = 0.4;
camlight;
camlight('headlight');
camlight('left');
lighting gouraud;
grid on;box on;
axis equal
alpha(0.9)


%% Plot energy components 
if storeValues
    figure('Position',[676   65  787  404],'color','w');hold all
    plot(t_store, Etot_store','-','LineWidth',2)
    plot(t_store, mu_store,'--','LineWidth',2)
    box on;
    set(gca,'LineWidth',1.5,'FontSize',14);
    
    l1 = legend('Kinetic (x)','Kinetic (y)','Kinetic (z)','Kinetic (total)','Potential','Interaction','Total ($E_{kin}+E_{pot}+E_{int}$)','Chemical Potential $\mu$');
    l1.Interpreter = 'Latex';
    l1.FontSize = 12;
    l1.Location = 'NorthEastOutside';
    xlabel('Imaginary Time  $\omega_x\tau$','Interpreter','Latex');ylabel('Energy per Particle / $\hbar\omega_x$','Interpreter','Latex')
    set(gca,'FontName','serif')
    
    plts = get(gca,'Children');
    CT1 = (cbrewer('qual', 'Dark2', length(plts),'spline'));
    for ii = 1:length(plts)
        set(plts(ii), 'color', CT1(ii,:))
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Subfunctions required for simulation %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x, kx] = grid_vectors(Nx, dx)
%%% Function for generating real space and frequency vectors in each dirn.

Fsx = 1/dx;         % Sampling frequency
dfx = Fsx/Nx;       % Frequency resolution

Nx_IS_EVEN =  ~rem(Nx,2);
if Nx_IS_EVEN; x  = -(Nx*dx/2):dx:(Nx*dx/2-dx); % Real space vector
               fx = -(Fsx/2):dfx:(Fsx/2-dfx);   % Spatial frequency vector
else;          x  = -((Nx-1)*dx/2):dx:((Nx-1)*dx/2);
               fx = -(Fsx/2-dfx/2):dfx:(Fsx/2-dfx/2);
end

kx = 2*pi*fx;       % Angular spatial frequency k vector

end





function [E, mu] = calculate_energy_and_mu(psi, dV, kx, ky, kz, V_ext, G)
%%% Calculate total energy of wavefunction (sum of kinetic, potential and
%%% interaction energies). Obtained using the expectation value of the
%%% Hamiltonian operator. Note, interaction energy includes a factor of
%%% 1/2 for double counting (see Dalfovo Rev. Mod. Phys 71 1999, Eq. 37),
%%% but the calculation of chemical potential does not.

%%% Kinetic Energy %%%%%%%%%%%%%%%%%%
psi_hat = fftn( ifftshift(psi) );
E_kin_x = 1/2 * sum(sum(sum(  abs( ifftn( 1i*kx.*psi_hat ) ).^2 )))*dV;
E_kin_y = 1/2 * sum(sum(sum(  abs( ifftn( 1i*ky.*psi_hat ) ).^2 )))*dV;
E_kin_z = 1/2 * sum(sum(sum(  abs( ifftn( 1i*kz.*psi_hat ) ).^2 )))*dV;
E_kin = E_kin_x + E_kin_y + E_kin_z;

%%% External Potential Energy %%%%%%%
psi_sq = abs(psi).^2;
E_pot = sum(sum(sum( V_ext.*psi_sq )))*dV;

%%% Mean Field Interaction Energy %%%
E_int = G/2 * sum(sum(sum( psi_sq.^2 )))*dV;

%%% Total Energy %%%%%%%%%%%%%%%%%%%%
E_tot = E_kin + E_pot + E_int;
E = [E_kin_x ; E_kin_y ; E_kin_z ; E_kin ; E_pot ; E_int ; E_tot];

%%% Chemical Potential %%%%%%%%%%%%%%
mu = E_kin + E_pot + 2*E_int;

end


function n_ThomasFermi_3D = ThomasFermi_3D_density(X, Y, Z, omega_x, omega_y, omega_z, N, c)
%%% Calculate 3D Thomas-Fermi limit (3D parabola)

mu_3D = (15*sqrt(2)/(32*pi)*N*c.g_int3D*c.mRb87^(3/2)*omega_x*omega_y*omega_z)^(2/5);
R_x = sqrt( 2*mu_3D/(c.mRb87*omega_x^2) );
R_y = sqrt( 2*mu_3D/(c.mRb87*omega_y^2) );
R_z = sqrt( 2*mu_3D/(c.mRb87*omega_z^2) );

n_ThomasFermi_3D = mu_3D/c.g_int3D * max( (1 - X.^2/R_x^2 - Y.^2/R_y^2 - Z.^2/R_z^2) , 0 );

end

function n_NonInteracting_3D = NonInteracting_3D_density(X, Y, Z, omega_x, omega_y, omega_z, N, c)
%%% Calculate 3D Non-Interacting limit (3D Gaussian)

ax = sqrt( c.hbar/(c.mRb87*omega_x) );
ay = sqrt( c.hbar/(c.mRb87*omega_y) );
az = sqrt( c.hbar/(c.mRb87*omega_z) );
psi_NonInteracting_3D = sqrt( 1/( pi^(3/2)*ax*ay*az ) ) * exp( -X.^2/(2*ax^2) - Y.^2/(2*ay^2) - Z.^2/(2*az^2) );
n_NonInteracting_3D = N * abs(psi_NonInteracting_3D).^2;

end