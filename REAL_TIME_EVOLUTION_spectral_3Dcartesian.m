
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------- Real Time GPE Evolution ---------------%
%-------- T. Barrett, Uni. of Sussex. 2021 -----------%
%-----------------------------------------------------%
% Calculates the real time GPE evolution for a BEC in %
% a given trap using the split-step Fourier spectral  %
% method, with second order Strang splitting, in 3D   %
% cartesian coordinates.                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc;close all;
disp([datestr(now), ' : ', ' Starting ',mfilename,'.m']);
addpath(genpath('Tools'))
addpath(genpath('outputs'))
c = physical_constants();     % Structure to store required physical constants
load('psi_ground.mat')        % Load the ground state (pre-calculated using imaginary time)
load FFT_prime_lengths.mat % Load optimium sizes for FFT speed

output_filename = "psi_tGPE.mat";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------ Spatial Grids -----------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nx = size(psi_k,2);         % Number of x grid points
Ny = size(psi_k,1);         % Number of y grid points
Nz = size(psi_k,3);         % Number of z grid points
dx = x(2) - x(1);           % Spacing of x grid
dy = y(2) - y(1);           % Spacing of y grid
dz = z(2) - z(1);           % Spacing of z grid
dV = dx*dy*dz;              % Volume element
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Input parameters in dimensionless units %%%
omega_x = 2*pi*20;     % x trap frequency in radians per second = 2*pi x Hz
l = sqrt( c.hbar/(c.mRb87*omega_x) );   % Unit of length ( x harmonic oscillator size )
N = 1e4;  % Number of particles in BEC
G = 4*pi*c.aRb87*N/l;                   % Scaled three-dimensional non-linearity

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------------- Numerical Switches ----------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UseGPU = 0;      % Use GPU for the calculation if one is available
updatePlots = 1; % Update plots periodically throughout simulation
storeValues = 1; % Store values (energy, mu, wavefunction cuts, etc)
includeFluctPhase = 0;
disable_x_expansion = 1;
apply_absorbing_boundaries = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------- Trap parameters -----------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Temperature = 230e-9;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------- Real Time Grid ------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_GPE = 3e-3;          % Time that the GPE evolution was run until (ms) 4.14495e-3
Nt = 2000;                    % Number of GPE evolution time iterations
dt = t_GPE*omega_x/(Nt-1);   % GPE evolution time step (dimensionless units)
T = (Nt-1)*dt;               % Duration of total time window
t = 0:dt:T;                  % Define time vector
fprintf('Final time = %0.5f ms \n', T/omega_x*1e3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------- Parameters for Expanding Grids ---------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dGridCheck = 10;                % Every dGridCheck steps check fraction of grid occupied by density distribution
gridExpansionThresh = 0.7;      % When wavefunction fills a certain fraction of the total grid, e.g one half
sigExpansionFac = 3;            % How many times sigma to trigger the grid expansion (e.g. 3 sigma encompasses almost all of the signal)
expand_fac = 1.3;               % How much to expand grid by (e.g. multiply number of grid points by 2, i.e. double)

if(apply_absorbing_boundaries)
    bnd_frac = 0.9;                 % Where the absorbing boundary is placed (e.g 0.7*xmax) - everything past this is multiplied by mask function
    [MSK_vec_x, MSK_vec_y, MSK_vec_z] = create_mask_vectors(x,y,z,Nx,Ny,Nz,bnd_frac);
    psi_k = psi_k.*MSK_vec_x.*MSK_vec_y.*MSK_vec_z;  % Apply mask function, to absorb reflections at grid boundaries
else
    bnd_frac = NaN;
    MSK_vec_x = NaN;
    MSK_vec_y = NaN;
    MSK_vec_z = NaN;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%% Add initial extra zero-padding %%%%%%%%%%
% (to allow some initial room for expansion compared to ground state grid)
Nx_new = 64;
Ny_new = 64;
Nz_new = 64;
assert( all([Nx_new Ny_new Nz_new] >= [Nx Ny Nz]), 'Padded initial grid sizes cannot be smaller than those of the ground state')

[x, kx] = grid_vectors(Nx_new, dx); % Create new grid vectors
[y, ky] = grid_vectors(Ny_new, dy); %
[z, kz] = grid_vectors(Nz_new, dz); %

% Permute variables to enable use of Matlab's implicit expansion
x = permute(x, [1 2 3]); % Creates a [1 x Nx x 1] vector
y = permute(y, [2 1 3]); % Creates a [Ny x 1 x 1] vector
z = permute(z, [3 1 2]); % Creates a [1 x 1 x Nz] vector

[~,x0_ind] = min(abs(x));  % Get indices of zero points for later slicing
[~,y0_ind] = min(abs(y));  % ( should be (N/2 + 1) if N is even )
[~,z0_ind] = min(abs(z));  %

% Pad the wavefunction with vacuum (zeros) to reach size of Nx_new x Ny_new x Nz_new
psi_k = padarray(psi_k, [Ny_new-Ny Nx_new-Nx Nz_new-Nz]/2, 0, 'both');

Nx = Nx_new;
Ny = Ny_new;
Nz = Nz_new;
clear Nx_new Ny_new Nz_new

% Regenerate mask function on newly padded grid
if(apply_absorbing_boundaries)
    [MSK_vec_x, MSK_vec_y, MSK_vec_z] = create_mask_vectors(x,y,z,Nx,Ny,Nz,bnd_frac);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%% Create k-vectors %%%%%%%%%%%%%%%%%
% Permute variables to enable use of Matlab's implicit expansion
kx = permute(ifftshift(kx), [1 2 3]); % Creates a [1 x Nx x 1] vector
ky = permute(ifftshift(ky), [2 1 3]); % Creates a [Ny x 1 x 1] vector
kz = permute(ifftshift(kz), [3 1 2]); % Creates a [1 x 1 x Nz] vector

Tx = exp( -1i*dt/2 * kx.^2); % Kinetic energy operators
Ty = exp( -1i*dt/2 * ky.^2);
Tz = exp( -1i*dt/2 * kz.^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%% External trapping potential %%%%%%%%%%%%
V_ext = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------ Standard deviation of initial density -----%
%-------- (to estimate its initial extent) --------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate standard deviations from second moments (variances)
% (Expectations of x_i^2, i.e. Var(x) = <(x - mu)^2> )
INT_psi = sum(sum(sum( abs( psi_k ).^2 ) ) ) * dV;
Sx = sqrt( sum(sum(sum( abs(psi_k).^2.*x.^2 )))*dV / INT_psi ) ;
Sy = sqrt( sum(sum(sum( abs(psi_k).^2.*y.^2 )))*dV / INT_psi ) ;
Sz = sqrt( sum(sum(sum( abs(psi_k).^2.*z.^2 )))*dV / INT_psi ) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%% Phase Fluctuations %%%%%%%%%%%%%%%%
if includeFluctPhase

    omega_ho = (omega_x*omega_y*omega_z)^(1/3);
    Tc = 0.94*c.hbar/c.kB*omega_ho*N^(1/3); % BEC transition temperature in harmonic trap
    fprintf('T/Tc = %.2f \n',Temperature/Tc)

    %%% Calculate phase according to OU method %%%%%%%%%%%%%%%%%%%%%%%%%%%
    oversample_factor = 20; % Factor by which to interpolate the phase profile
    sigma_conv = 0.5e-6;    % Width of Gaussian point spread function [m]
    density_thresh = 0.01; % Do not calculate phase where there is too little density
    phi_OU = generate_OU_phase(psi_k, x, y, z, l, N, Temperature, density_thresh, oversample_factor, sigma_conv);
    %     psi_k = sqrt( abs(psi_k).^2 ); % Reset ground state phase to zero
    psi_k = psi_k .* exp(1i*phi_OU); % Imprint phase on wavefunction

    figure;
    subplot(3,1,1)
    imagesc(x*l*1e6,y*l*1e6, abs(psi_k(:,:,z0_ind)).^2  )
    subplot(3,1,2)
    imagesc(x*l*1e6,y*l*1e6, angle(psi_k(:,:,z0_ind)),[-1 1] )
    subplot(3,1,3)
    plot( x*l*1e6, squeeze((angle(psi_k(y0_ind,:,z0_ind)))))
end

psihat_k = fftshift( fftn( ifftshift(psi_k) ) )*dV;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------ Set Up Plots ------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if updatePlots
    h.f1 = figure('Position',[0.0338    0.0546    1.4592    0.7074]*1e3);

    subplot(3,3,1);hold all;
    h.p1 = plot( squeeze(x), squeeze( abs(psi_k(y0_ind,:,z0_ind)).^2 ) /(l^3)*N/(1e18) ,'o-');
    xlabel('x');ylabel('Density (\mum^-^3)');
    set(gca,'FontSize',14,'Box','on')
    h.l1a = line([1 1]*sigExpansionFac*Sx,[0 1000],'LineStyle','--','color','b');
    h.l1b = line([1 1]*x(Nx)*gridExpansionThresh,[0 1000],'LineStyle','--','color','r');
    h.l1c = line([1 1]*x(Nx)*bnd_frac,[0 1000],'LineStyle','--','color','g');
    h.ax(1) = gca;

    subplot(3,3,2);hold all;
    h.p2 = plot( squeeze(y), squeeze( abs(psi_k(:,x0_ind,z0_ind)).^2 ) /(l^3)*N/(1e18) ,'o-');
    xlabel('y');ylabel('Density (\mum^-^3)');
    set(gca,'FontSize',14,'Box','on')
    h.l2a = line([1 1]*sigExpansionFac*Sy,[0 1000],'LineStyle','--','color','b');
    h.l2b = line([1 1]*y(Ny)*gridExpansionThresh,[0 1000],'LineStyle','--','color','r');
    h.l2c = line([1 1]*y(Ny)*bnd_frac,[0 1000],'LineStyle','--','color','g');
    h.ax(2) = gca;

    subplot(3,3,3);hold all;
    h.p3 = plot( squeeze(z), squeeze( abs(psi_k(y0_ind,x0_ind,:)).^2 ) /(l^3)*N/(1e18) ,'o-');
    xlabel('z');ylabel('Density (\mum^-^3)');
    set(gca,'FontSize',14,'Box','on')
    h.l3a = line([1 1]*sigExpansionFac*Sz,[0 1000],'LineStyle','--','color','b');
    h.l3b = line([1 1]*z(Nz)*gridExpansionThresh,[0 1000],'LineStyle','--','color','r');
    h.l3c = line([1 1]*z(Nz)*bnd_frac,[0 1000],'LineStyle','--','color','g');
    h.ax(3) = gca;

    subplot(3,3,4);
    h.p4 = plot( squeeze(x), squeeze( angle(psi_k(y0_ind,:,z0_ind)) ) ,'o-');
    xlabel('x');ylabel('Phase');
    set(gca,'FontSize',14,'Box','on')
    h.ax(4) = gca;

    subplot(3,3,5);
    h.p5 = plot( squeeze(y), squeeze( angle(psi_k(:,x0_ind,z0_ind)) ) ,'o-');
    xlabel('y');ylabel('Phase');
    set(gca,'FontSize',14,'Box','on')
    h.ax(5) = gca;

    subplot(3,3,6);
    h.p6 = plot( squeeze(z), squeeze( angle(psi_k(y0_ind,x0_ind,:)) ) ,'o-');
    ylim([-50 50])
    xlabel('z');ylabel('Phase');
    set(gca,'FontSize',14,'Box','on')
    h.ax(6) = gca;

    % Plot in k-space
    subplot(3,3,7);
    h.p7 = plot( fftshift(squeeze( kx ) ),  real( squeeze(psihat_k(y0_ind,:,z0_ind)) ) ,'o-');
    xlabel('k_x');ylabel('Re(\psi(k))');set(gca,'FontSize',14,'Box','on')
    subplot(3,3,8);
    h.p8 = plot( fftshift(squeeze( ky ) ),  real( squeeze(psihat_k(:,x0_ind,z0_ind)) ) ,'o-');
    xlabel('k_y');ylabel('Re(\psi(k))');set(gca,'FontSize',14,'Box','on')
    subplot(3,3,9);
    h.p9 = plot( fftshift(squeeze( kz ) ),  real( squeeze(psihat_k(y0_ind,x0_ind,:)) ) ,'o-');
    xlabel('k_z');ylabel('Re(\psi(k))');set(gca,'FontSize',14,'Box','on')

    number_3D_plots = 4;
    h.f3 = figure('color','w','Position',[488 62  588  796]);
    plt = 1;
    subplot(number_3D_plots,1,plt)
    s(plt) = isosurface(x*l*1e6,y*l*1e6,z*l*1e6, abs(psi_k).^2 ,max(max(max(abs(psi_k).^2)))*0.03);
    p(plt) = patch(s(plt));
    isonormals(x*l*1e6,y*l*1e6,z*l*1e6, abs(psi_k).^2 ,p(plt))
    view(3);
    set(p(plt),'FaceColor',[33,113,181]/255);
    set(p(plt),'EdgeColor','none');
    p(plt).SpecularStrength = 0.4;
    camlight;
    camlight('headlight');
    camlight('left');
    lighting gouraud;
    axis equal
    alpha(0.75);
    title( sprintf('t = %.1f ms', 0 ) )
    xlabel('$x$ ($\mu$m)','Interpreter','Latex');
    ylabel('$y$ ($\mu$m)','Interpreter','Latex');
    zlabel('$z$ ($\mu$m)','Interpreter','Latex');
    set(gca,'FontSize',12)
    grid on;box on;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-------- Set up storage vectors/matrices ---------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dStore = 20;     % Only store every 'dStore' iterations
store_ind_plt_list = round(Nt/dStore/(number_3D_plots-1))*[1:(number_3D_plots-2)];
if storeValues
    num_to_store = ceil(Nt/dStore);              % Total number of iterations that will be stored
    t_store = zeros( 1, num_to_store );          % Preallocate storage arrays
    Etot_store = zeros( 7, num_to_store );       % Preallocate storage arrays
    mu_store = zeros( 1, num_to_store );         % Preallocate storage arrays
    psi_store = struct('x',           cell(1, num_to_store),...
        'y',           cell(1, num_to_store),...
        'z',           cell(1, num_to_store),...
        'psi_xcut',    cell(1, num_to_store),...
        'psi_ycut',    cell(1, num_to_store),...
        'psi_zcut',    cell(1, num_to_store),...
        'psi_xz_slice',cell(1, num_to_store),...
        'n2D_xz_int',  cell(1, num_to_store),...
        'kx',          cell(1, num_to_store),...
        'ky',          cell(1, num_to_store),...
        'kz',          cell(1, num_to_store),...
        'psihat_xcut', cell(1, num_to_store),...
        'psihat_ycut', cell(1, num_to_store),...
        'psihat_zcut', cell(1, num_to_store)...
        );

    %%% Store the first value in each storage matrix %%%
    store_ind = 1;
    t_store(store_ind) = t(1);
    INT_psi = sum(sum(sum( abs(psi_k).^2 )))*dV; % Calculate the integral
    [Etot_store(:,store_ind), mu_store(store_ind)] = calculate_energy_and_mu(psi_k, dV, kx, ky, kz, V_ext, G, INT_psi);
    psi_store(store_ind).x = squeeze(x);
    psi_store(store_ind).y = squeeze(y);
    psi_store(store_ind).z = squeeze(z);
    psi_store(store_ind).psi_xcut = squeeze( psi_k(y0_ind,:,z0_ind) );
    psi_store(store_ind).psi_ycut = squeeze( psi_k(:,x0_ind,z0_ind) );
    psi_store(store_ind).psi_zcut = squeeze( psi_k(y0_ind,x0_ind,:) );
    psi_store(store_ind).psi_xz_slice = squeeze( psi_k(y0_ind,:,:) );
    psi_store(store_ind).n2D_xz_int = squeeze( sum(abs(psi_k).^2,1)*dy );
    psi_store(store_ind).kx = fftshift(squeeze( kx ) );
    psi_store(store_ind).ky = fftshift(squeeze( ky ) );
    psi_store(store_ind).kz = fftshift(squeeze( kz ) );
    psi_store(store_ind).psihat_xcut = squeeze( psihat_k(y0_ind,:,z0_ind) );
    psi_store(store_ind).psihat_ycut = squeeze( psihat_k(:,x0_ind,z0_ind) );
    psi_store(store_ind).psihat_zcut = squeeze( psihat_k(y0_ind,x0_ind,:) );

    disp('--------------')
    fprintf('Storing...  iteration 1 of %i: Energy = %0.10f, mu = %0.10f \n', Nt, Etot_store(7,store_ind), mu_store(store_ind));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n');
fprintf('=============================================================\n');
fprintf('---------- Initial Energy Components per Particle -----------\n');
fprintf('-------------------- (in units of ℏ*ωx) ---------------------\n');
fprintf('=============================================================\n');
fprintf('Kinetic (x-direction)  = %0.6f \n', Etot_store(1,1));
fprintf('Kinetic (y-direction)  = %0.6f \n', Etot_store(2,1));
fprintf('Kinetic (z-direction)  = %0.6f \n', Etot_store(3,1));
fprintf('Kinetic (total)        = %0.6f \n', Etot_store(4,1));
fprintf('Potential              = %0.6f \n', Etot_store(5,1));
fprintf('Interaction            = %0.6f \n', Etot_store(6,1));
fprintf('Total                  = %0.6f \n', Etot_store(7,1));
fprintf('Chemical Potential     = %0.6f \n', mu_store(1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------ Move arrays to GPU memory -----------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(UseGPU)
    try
        gpuArray(1);
        UseGPU = 1;
        fprintf('GPU device available to use.\n')
        gpu = gpuDevice;
        fprintf('Total GPU memory:       %0.3f GB \n', gpu.TotalMemory / 1e9);
        fprintf('Free GPU memory:        %0.3f GB \n', gpu.FreeMemory / 1e9);
        fprintf('Available GPU memory:   %0.3f GB \n', gpu.AvailableMemory / 1e9);
    catch ME
        UseGPU = 0;
        fprintf('GPU device not available to use.\n')
    end
end

if UseGPU
    fprintf('\nMoving wavefunction to GPU... \n')
    initialGPUmemoryAvailable = gpu.AvailableMemory;
    GPU_memorystatus(1).WavefunctionSize = numel(psi_k)*16;
    fprintf('Wavefunction size:      %0.3f GB \n', GPU_memorystatus(1).WavefunctionSize / 1e9);
    psi_k = gpuArray(psi_k); % Iterated wavefunction
    V_ext = gpuArray(V_ext); % External potential matrix
    Tx = gpuArray(Tx);       % Kinetic energy operator in x
    Ty = gpuArray(Ty);       % Kinetic energy operator in y
    Tz = gpuArray(Tz);       % Kinetic energy operator in z

    if(apply_absorbing_boundaries)
        MSK_vec_x = gpuArray(MSK_vec_x);
        MSK_vec_y = gpuArray(MSK_vec_y);
        MSK_vec_z = gpuArray(MSK_vec_z);
    end

    GPU_memorystatus(1).FreeMemory = gpu.FreeMemory;
    GPU_memorystatus(1).AvailableMemory = gpu.AvailableMemory;
    fprintf('Free GPU memory:        %0.3f GB \n',  GPU_memorystatus(1).FreeMemory / 1e9);
    fprintf('Available GPU memory:   %0.3f GB \n',  GPU_memorystatus(1).AvailableMemory / 1e9);
    fprintf('GPU memory used:        %0.3f GB \n', (initialGPUmemoryAvailable - GPU_memorystatus(1).AvailableMemory) / 1e9 );
    GPUmemory_ind = 2;

    h.f2 = figure;hold all;
    h.p4 = plot( [GPU_memorystatus.FreeMemory] / 1e9       ,'o-' );
    h.p5 = plot( [GPU_memorystatus.AvailableMemory] / 1e9  ,'o-' );
    h.p6 = plot( [GPU_memorystatus.WavefunctionSize] / 1e9 ,'o-' );
    legend('Free Memory','Available Memory','Wavefunction Size','location','SouthWest')
    ylabel('Memory (GB)')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------ Main Time Stepping Loop -------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pause(1)
tic
fprintf('Running real time evolution... \n');
% If we are not interested in storing/plotting, then the two half steps
% which appear in the loop can be combined into one full step, speeding up
% the code. In this case, we need to do an extra half step before the loop.
if ~updatePlots && ~storeValues
    psi_k = exp( -1i*dt/2*(  V_ext + G*abs(psi_k).^2  ) ).*psi_k;         % Half step with potential operator
    Nt = Nt-1;
end

if(UseGPU)
    GPU_memorystatus(GPUmemory_ind).WavefunctionSize = numel(psi_k)*16;
    GPU_memorystatus(GPUmemory_ind).FreeMemory = gpu.FreeMemory;
    GPU_memorystatus(GPUmemory_ind).AvailableMemory = gpu.AvailableMemory;
    fprintf('Wavefunction size:      %0.3f GB \n', GPU_memorystatus(GPUmemory_ind).WavefunctionSize / 1e9);
    fprintf('Free GPU memory:        %0.3f GB \n',  GPU_memorystatus(GPUmemory_ind).FreeMemory / 1e9);
    fprintf('Available GPU memory:   %0.3f GB \n',  GPU_memorystatus(GPUmemory_ind).AvailableMemory / 1e9);
    fprintf('GPU memory used:        %0.3f GB \n', (initialGPUmemoryAvailable - GPU_memorystatus(GPUmemory_ind).AvailableMemory) / 1e9 );
    GPUmemory_ind = GPUmemory_ind + 1;
end

for k = 2:Nt

    %%%%%%% Check if grids need to be expanded %%%%%%%%
    if mod(k,dGridCheck) == 1 || dGridCheck == 1
        INT_psi = sum(sum(sum( abs( psi_k ).^2 ) ) ) * dV;
        Sx = sqrt( sum(sum(sum( abs(psi_k).^2.*x.^2 )))*dV  / INT_psi ) ; % RMS width in x
        Sy = sqrt( sum(sum(sum( abs(psi_k).^2.*y.^2 )))*dV  / INT_psi ) ; % RMS width in y
        Sz = sqrt( sum(sum(sum( abs(psi_k).^2.*z.^2 )))*dV  / INT_psi ) ; % RMS width in z

        DirToExpand = [Sx Sy Sz]*sigExpansionFac > [x(Nx) y(Ny) z(Nz)]*gridExpansionThresh; % Check cloud size against threshold

        if disable_x_expansion
            DirToExpand(1) = 0;
        end

        if any( DirToExpand ) % If any of the directions do indeed need expanding

            if(UseGPU)
                % Track memory usage before expanding grids
                GPU_memorystatus(GPUmemory_ind).WavefunctionSize = numel(psi_k)*16;
                GPU_memorystatus(GPUmemory_ind).FreeMemory = gpu.FreeMemory;
                GPU_memorystatus(GPUmemory_ind).AvailableMemory = gpu.AvailableMemory;
                fprintf('Wavefunction size:      %0.3f GB \n', GPU_memorystatus(GPUmemory_ind).WavefunctionSize / 1e9);
                fprintf('Free GPU memory:        %0.3f GB \n',  GPU_memorystatus(GPUmemory_ind).FreeMemory / 1e9);
                fprintf('Available GPU memory:   %0.3f GB \n',  GPU_memorystatus(GPUmemory_ind).AvailableMemory / 1e9);
                fprintf('GPU memory used:        %0.3f GB \n', (initialGPUmemoryAvailable - GPU_memorystatus(GPUmemory_ind).AvailableMemory) / 1e9 );
                GPUmemory_ind = GPUmemory_ind + 1;

                psi_k = gather(psi_k); % Bring current wavefunction back from GPU memory
                V_ext = gather(V_ext);
            end

            [x,y,z,kx,ky,kz,Tx,Ty,Tz,Nx,Ny,Nz,x0_ind,y0_ind,z0_ind,psi_k] = expand_grids(DirToExpand, ...
                expand_fac, FFT_prime_lengths,dt, Nx, Ny, Nz, dx, dy, dz, psi_k);

            if(apply_absorbing_boundaries)
                [MSK_vec_x, MSK_vec_y, MSK_vec_z] = create_mask_vectors(x,y,z,Nx,Ny,Nz,bnd_frac);
            end

            % Move new variables to GPU
            if UseGPU
                reset(gpu)
                psi_k = gpuArray(psi_k); % Take current wavefunction back to GPU memory

                Tx = gpuArray(Tx);       % Kinetic energy operator in x
                Ty = gpuArray(Ty);       % Kinetic energy operator in y
                Tz = gpuArray(Tz);       % Kinetic energy operator in z
                if(apply_absorbing_boundaries)
                    MSK_vec_x = gpuArray(MSK_vec_x);
                    MSK_vec_y = gpuArray(MSK_vec_y);
                    MSK_vec_z = gpuArray(MSK_vec_z);
                end
                V_ext = gpuArray(V_ext); % External potential matrix

                % Track memory usage after expanding grids
                GPU_memorystatus(GPUmemory_ind).WavefunctionSize = numel(psi_k)*16;
                GPU_memorystatus(GPUmemory_ind).FreeMemory = gpu.FreeMemory;
                GPU_memorystatus(GPUmemory_ind).AvailableMemory = gpu.AvailableMemory;
                fprintf('Wavefunction size:      %0.3f GB \n', GPU_memorystatus(GPUmemory_ind).WavefunctionSize / 1e9);
                fprintf('Free GPU memory:        %0.3f GB \n',  GPU_memorystatus(GPUmemory_ind).FreeMemory / 1e9);
                fprintf('Available GPU memory:   %0.3f GB \n',  GPU_memorystatus(GPUmemory_ind).AvailableMemory / 1e9);
                fprintf('GPU memory used:        %0.3f GB \n', (initialGPUmemoryAvailable - GPU_memorystatus(GPUmemory_ind).AvailableMemory) / 1e9 );
                GPUmemory_ind = GPUmemory_ind + 1;
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%% Split-Step Spectral Evolution (using Strang Splitting) %%%
    %%% (combine half steps into a single full step if not plotting/storing)
    if ~updatePlots && ~storeValues
        psi_k = fftshift( ifftn( Tx.*Ty.*Tz.*fftn( ifftshift(psi_k) ) ) ); % Full step with kinetic operator (using implicit expansion)
        psi_k = exp( -1i*dt*(  V_ext + G*abs(psi_k).^2  ) ).*psi_k;        % Full step with potential operator
    else
        psi_k = exp( -1i*dt/2*(  V_ext + G*abs(psi_k).^2  ) ).*psi_k;      % Half step with potential operator
        psi_k = fftshift( ifftn( Tx.*Ty.*Tz.*fftn( ifftshift(psi_k) ) ) ); % Full step with kinetic operator (using implicit expansion)
        psi_k = exp( -1i*dt/2*(  V_ext + G*abs(psi_k).^2  ) ).*psi_k;      % Half step with potential operator
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if(apply_absorbing_boundaries)
        psi_k = psi_k.*MSK_vec_x.*MSK_vec_y.*MSK_vec_z;  % Apply mask function, to absorb reflections at grid boundaries
    end

    %%% Store Values Every dStore Iterations %%%%%%%%%%%%%%%%%%%
    if (mod(k,dStore) == 1 || dStore == 1) && storeValues

        store_ind = store_ind + 1;
        t_store(store_ind) = t(k);

        % Calculate energy and mu (on GPU if available)
        INT_psi = sum(sum(sum( abs(psi_k).^2 )))*dV; % Calculate the integral
        [E, mu] = calculate_energy_and_mu(psi_k, dV, kx, ky, kz, V_ext, G, INT_psi);
        fprintf('Storing...  iteration %i of %i: Energy = %0.10f, mu = %0.10f, norm = %0.10f \n', k, Nt, E(7),mu,INT_psi);
        Etot_store(:,store_ind) = E;
        mu_store(:,store_ind) = mu;

        % Calculate k-space wavefunction (on GPU if available)
        psihat_k = fftshift( fftn( ifftshift(psi_k) ) )*dV;

        if UseGPU
            psi_k = gather(psi_k); % Bring current wavefunction back from GPU memory
            psihat_k = gather(psihat_k); % Bring k-space wavefunction back from GPU memory
        end

        % Store the real-space wavefunction line cuts
        psi_store(store_ind).x = squeeze(x);
        psi_store(store_ind).y = squeeze(y);
        psi_store(store_ind).z = squeeze(z);
        psi_store(store_ind).psi_xcut = squeeze( psi_k(y0_ind,:,z0_ind) );
        psi_store(store_ind).psi_ycut = squeeze( psi_k(:,x0_ind,z0_ind) );
        psi_store(store_ind).psi_zcut = squeeze( psi_k(y0_ind,x0_ind,:) );

        % Store the 2D planes
        psi_store(store_ind).psi_xz_slice = squeeze( psi_k(y0_ind,:,:) );
        psi_store(store_ind).n2D_xz_int = squeeze( sum( abs(psi_k).^2 ,1)*dy );

        % Store the k-space wavefunction cuts
        psi_store(store_ind).kx = fftshift(squeeze( kx ) );
        psi_store(store_ind).ky = fftshift(squeeze( ky ) );
        psi_store(store_ind).kz = fftshift(squeeze( kz ) );

        psi_store(store_ind).psihat_xcut = squeeze( psihat_k(y0_ind,:,z0_ind) );
        psi_store(store_ind).psihat_ycut = squeeze( psihat_k(:,x0_ind,z0_ind) );
        psi_store(store_ind).psihat_zcut = squeeze( psihat_k(y0_ind,x0_ind,:) );

        %%% Plot isosurfaces of the density during simulation if required
        %%% (plotting in this way doesn't require storing multiple 3D matrices)
        if ismember(store_ind,store_ind_plt_list)
            plt = plt + 1;
            figure(h.f3);
            subplot(number_3D_plots,1,plt)
            s(plt) = isosurface(x*l*1e6,y*l*1e6,z*l*1e6,abs(psi_k).^2,max(max(max(abs(psi_k).^2)))*0.03);
            p(plt) = patch(s(plt));
            isonormals(x*l*1e6,y*l*1e6,z*l*1e6,abs(psi_k).^2,p(plt))
            view(3);
            set(p(plt),'FaceColor',[33,113,181]/255);
            set(p(plt),'EdgeColor','none');
            p(plt).SpecularStrength = 0.4;
            camlight;
            camlight('headlight');
            camlight('left');
            lighting gouraud;
            axis equal
            alpha(0.75);
            title( sprintf('t = %.1f ms', t_store(store_ind)/omega_x*1e3) )
            xlabel('$x$ ($\mu$m)','Interpreter','Latex');
            ylabel('$y$ ($\mu$m)','Interpreter','Latex');
            zlabel('$z$ ($\mu$m)','Interpreter','Latex');
            set(gca,'FontSize',12)
            grid on;box on;
        end

        if UseGPU
            psi_k = gpuArray(psi_k); % Take current wavefunction back to GPU memory
        end

        % Track memory usage
        if UseGPU
            GPU_memorystatus(GPUmemory_ind).WavefunctionSize = numel(psi_k)*16;
            GPU_memorystatus(GPUmemory_ind).FreeMemory = gpu.FreeMemory;
            GPU_memorystatus(GPUmemory_ind).AvailableMemory = gpu.AvailableMemory;
            %         fprintf('Wavefunction size:      %0.3f GB \n', GPU_memorystatus(GPUmemory_ind).WavefunctionSize / 1e9);
            %         fprintf('Free GPU memory:        %0.3f GB \n',  GPU_memorystatus(GPUmemory_ind).FreeMemory / 1e9);
            %         fprintf('Available GPU memory:   %0.3f GB \n',  GPU_memorystatus(GPUmemory_ind).AvailableMemory / 1e9);
            %         fprintf('GPU memory used:        %0.3f GB \n', (initialGPUmemoryAvailable - GPU_memorystatus(GPUmemory_ind).AvailableMemory) / 1e9 );
            GPUmemory_ind = GPUmemory_ind + 1;
        end

    end

    %%% Display percentage completed and update plots %%%%%%%%%%%%%%%%%%%%%
    if (mod(k,floor(Nt/10)) == 0)

        fprintf('Iteration %i of %i (%.0f%% complete)\n', k, Nt, k/Nt*100);

        if updatePlots

            if UseGPU
                psi_k = gather(psi_k); % Bring current wavefunction back from GPU memory
            end

            % Update plots
            set(h.p1,'XData',squeeze(x),'YData', squeeze( abs(psi_k(y0_ind,:,z0_ind)).^2 ) /(l^3)/(1e18) );
            set(h.p2,'XData',squeeze(y),'YData', squeeze( abs(psi_k(:,x0_ind,z0_ind)).^2 ) /(l^3)/(1e18) );
            set(h.p3,'XData',squeeze(z),'YData', squeeze( abs(psi_k(y0_ind,x0_ind,:)).^2 ) /(l^3)/(1e18) );

            psihat_k = fftshift( fftn( ifftshift(psi_k) ) )*dV;        % Bring wavefunction into k-space
            set(h.p7,'XData', fftshift( squeeze(kx) ), 'YData', real( squeeze(psihat_k(y0_ind,:,z0_ind)) ) );
            set(h.p8,'XData', fftshift( squeeze(ky) ), 'YData', real( squeeze(psihat_k(:,x0_ind,z0_ind)) ) );
            set(h.p9,'XData', fftshift( squeeze(kz) ), 'YData', real( squeeze(psihat_k(y0_ind,x0_ind,:)) ) );

            INT_psi = sum(sum(sum( abs( psi_k ).^2 ) ) ) * dV;
            Sx = sqrt( sum(sum(sum( abs(psi_k).^2.*x.^2 )))*dV  / INT_psi ) ; % RMS width in x
            Sy = sqrt( sum(sum(sum( abs(psi_k).^2.*y.^2 )))*dV  / INT_psi ) ; % RMS width in y
            Sz = sqrt( sum(sum(sum( abs(psi_k).^2.*z.^2 )))*dV  / INT_psi ) ; % RMS width in z

            plt_ylims = [0 abs(psi_k(y0_ind,x0_ind,z0_ind)).^2 * 1.1 /(l^3)/(1e18)];
            set(h.l1a,'XData',[1 1]*sigExpansionFac*Sx , 'YData', plt_ylims)
            set(h.l2a,'XData',[1 1]*sigExpansionFac*Sy , 'YData', plt_ylims)
            set(h.l3a,'XData',[1 1]*sigExpansionFac*Sz , 'YData', plt_ylims)
            set(h.l1b,'XData',[1 1]*x(Nx)*gridExpansionThresh , 'YData', plt_ylims)
            set(h.l2b,'XData',[1 1]*y(Ny)*gridExpansionThresh , 'YData', plt_ylims)
            set(h.l3b,'XData',[1 1]*z(Nz)*gridExpansionThresh , 'YData', plt_ylims)
            set(h.l1c,'XData',[1 1]*x(Nx)*bnd_frac,'YData', plt_ylims);
            set(h.l2c,'XData',[1 1]*y(Ny)*bnd_frac,'YData', plt_ylims);
            set(h.l3c,'XData',[1 1]*z(Nz)*bnd_frac,'YData', plt_ylims);

            set( h.ax(1:3), 'YLim', plt_ylims)

            %set(p100, 'XData', x*l*1e6, 'YData', sum( abs(psi_k).^2 / (l^3) , [1 3]) * dy*l * dz*l ) ;

            if UseGPU
                set(h.p4 , 'YData' , [GPU_memorystatus.FreeMemory]       / 1e9)
                set(h.p5 , 'YData' , [GPU_memorystatus.AvailableMemory]  / 1e9)
                set(h.p6 , 'YData' , [GPU_memorystatus.WavefunctionSize] / 1e9)
            end


            drawnow

            if UseGPU
                psi_k = gpuArray(psi_k); % Take current wavefunction back to GPU memory
            end

        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
% If we are not interested in storing/plotting, then the two half steps
% which appear in the loop can be combined into one full step, speeding up
% the code. In this case, we need to do an extra half step after the loop.
if ~updatePlots && ~storeValues
    psi_k = fftshift( ifftn( Tx.*Ty.*Tz.*fftn( ifftshift(psi_k) ) ) ); % Full step with kinetic operator (using implicit expansion)
    psi_k = exp( -1i*dt/2*(  V_ext + G*abs(psi_k).^2  ) ).*psi_k;      % Half step with potential operator
end
toc

%%% Values from the final iteration %%%
INT_psi_final = sum(sum(sum( abs(psi_k).^2 )))*dV; % Calculate the integral
[E, mu] = calculate_energy_and_mu(psi_k, dV, kx, ky, kz, V_ext, G, INT_psi_final);
E_final = E;
mu_final = mu;

if UseGPU
    psi_k = gather(psi_k);
    E_final = gather(E_final);
    mu_final = gather(mu_final);
end

fprintf('\n');
fprintf('=============================================================\n');
fprintf('----------- Final Energy Components per Particle ------------\n');
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


%%% Save final state %%%%%%%%%%%%%%%%%
save('outputs\' + output_filename,'psi_k','x','y','z')

if includeFluctPhase
    save phase phi_OU
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Plot the stored energy components
if storeValues

    % Append final values to storage vectors
    if t_store(end) ~= t(end)
        num_to_store = num_to_store + 1;
        store_ind = num_to_store;
        t_store = [t_store t(end)];
        Etot_store = [Etot_store E_final];
        mu_store = [mu_store mu_final];

        % Store the real-space wavefunction line cuts
        psi_store(store_ind).x = squeeze(x);
        psi_store(store_ind).y = squeeze(y);
        psi_store(store_ind).z = squeeze(z);
        psi_store(store_ind).psi_xcut = squeeze( psi_k(y0_ind,:,z0_ind) );
        psi_store(store_ind).psi_ycut = squeeze( psi_k(:,x0_ind,z0_ind) );
        psi_store(store_ind).psi_zcut = squeeze( psi_k(y0_ind,x0_ind,:) );

        % Store the 2D planes
        psi_store(store_ind).psi_xz_slice = squeeze( psi_k(y0_ind,:,:) );
        psi_store(store_ind).n2D_xz_int = squeeze( sum( abs(psi_k).^2 ,1)*dy );

        % Store the k-space wavefunction cuts
        psi_store(store_ind).kx = fftshift(squeeze( kx ) );
        psi_store(store_ind).ky = fftshift(squeeze( ky ) );
        psi_store(store_ind).kz = fftshift(squeeze( kz ) );
        psihat_k = fftshift( fftn( ifftshift(psi_k) ) )*dV;
        psi_store(store_ind).psihat_xcut = squeeze( psihat_k(y0_ind,:,z0_ind) );
        psi_store(store_ind).psihat_ycut = squeeze( psihat_k(:,x0_ind,z0_ind) );
        psi_store(store_ind).psihat_zcut = squeeze( psihat_k(y0_ind,x0_ind,:) );

    end

    figure('Position',[261.0000  357.8000  787.0000  404.2000],'color','w');hold all
    plot(t_store/omega_x*1e3, Etot_store','-','LineWidth',2)
    plot(t_store/omega_x*1e3, mu_store,'--','LineWidth',2)
    box on;
    set(gca,'LineWidth',1.5,'FontSize',14);
    %set(gca,'Xlim',[0 0.2],'Ylim',[-0.8 100])

    l1 = legend('Kinetic (x)','Kinetic (y)','Kinetic (z)','Kinetic (total)','Potential','Interaction','Total ($E_{kin}+E_{pot}+E_{int}$)','Chemical Potential $\mu$');
    l1.Interpreter = 'Latex';
    l1.FontSize = 12;
    l1.Location = 'NorthEastOutside';
    xlabel('Time  $t$ (ms)','Interpreter','Latex');ylabel('Energy per Particle / $\hbar\omega_x$','Interpreter','Latex')
    set(gca,'FontName','serif')

    plts = get(gca,'Children');
    CT1 = (cbrewer('qual', 'Dark2', length(plts)));
    for ii = 1:length(plts)
        set(plts(ii), 'color', CT1(ii,:))
    end

    %%% Plot density line cut over time %%%
    M = zeros(num_to_store,Nz);
    for ii = 1:num_to_store
        M(ii,:) = abs( padarray( sum(psi_store(ii).n2D_xz_int,1)*dx, [0 Nz-size(psi_store(ii).n2D_xz_int,2)]/2, NaN, 'both') ).^2;
        M(ii,:) = M(ii,:)/max(M(ii,:));
    end
    M = M.';
    imAlpha=ones(size(M));
    imAlpha(isnan(M))=0;
    figure('color','w');
    imagesc(t_store/omega_x*1e3,psi_store(end).z*l*1e6,M,'AlphaData',imAlpha);
    colormap(viridis(100))
    colorbar
    xlabel('Time  $t$ (ms)','Interpreter','Latex');
    ylabel('$z$ ($\mu$m)','Interpreter','Latex');
    title('Normalised Integrated Density n(z)','Interpreter','Latex')
    set(gca,'color',0*[1 1 1],'YDir','normal','FontSize',12,'FontName','serif');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

figure('color','w','Position',[488 62  588  796])
%%% Plot initial iteration %%%
plt2 = 1;
subplot(number_3D_plots,1,1)
imagesc(psi_store(end).x*l*1e6,psi_store(end).z*l*1e6,padarray(psi_store(1).n2D_xz_int, [0 Nz-size(psi_store(1).z,1)]/2, NaN, 'both').')
axis equal tight
xlabel('x (\mum)');ylabel('z (\mum)');
title('t = 0.0 ms')
colorbar
%%% Plot iterations throughout evolution %%%
for k = 1:length(store_ind_plt_list)
    plt2 = plt2+1;
    subplot(number_3D_plots,1,plt2)
    imagesc(psi_store(end).x*l*1e6,psi_store(end).z*l*1e6,padarray(psi_store(store_ind_plt_list(k)).n2D_xz_int, [0 Nz-size(psi_store(store_ind_plt_list(k)).z,1)]/2, NaN, 'both').')
    axis equal tight
    xlabel('x (\mum)');ylabel('z (\mum)');
    title( sprintf('t = %.1f ms', t_store(store_ind_plt_list(k))/omega_x*1e3) )
    colorbar
end
%%% Plot final iteration %%%
plt2 = plt2+1;
subplot(number_3D_plots,1,plt2)
imagesc(psi_store(end).x*l*1e6,psi_store(end).z*l*1e6,psi_store(end).n2D_xz_int.')
axis equal tight
xlabel('x (\mum)');ylabel('z (\mum)');
title( sprintf('t = %.1f ms', t_store(end)/omega_x*1e3) )
colorbar
colormap(viridis(100))


plt = plt + 1;
figure(h.f3);
subplot(number_3D_plots,1,plt)
s(plt) = isosurface(x*l*1e6,y*l*1e6,z*l*1e6,abs(psi_k).^2,max(max(max(abs(psi_k).^2)))*0.03);
p(plt) = patch(s(plt));
isonormals(x*l*1e6,y*l*1e6,z*l*1e6,abs(psi_k).^2,p(plt))
view(3);
set(p(plt),'FaceColor',[33,113,181]/255);
set(p(plt),'EdgeColor','none');
p(plt).SpecularStrength = 0.4;
camlight;
camlight('headlight');
camlight('left');
lighting gouraud;
axis equal
alpha(0.75);
title( sprintf('t = %.1f ms', t_store(end)/omega_x*1e3) )
xlabel('$x$ ($\mu$m)','Interpreter','Latex');
ylabel('$y$ ($\mu$m)','Interpreter','Latex');
zlabel('$z$ ($\mu$m)','Interpreter','Latex');
set(gca,'FontSize',12)
grid on;box on;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Subfunctions required for simulation %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


% function [E, mu] = calculate_energy_and_mu(psi, dV, kx, ky, kz, V_ext, G)
% %%% Calculate total energy of wavefunction (sum of kinetic, potential and
% %%% interaction energies). Obtained using the expectation value of the
% %%% Hamiltonian operator. Note, interaction energy includes a factor of
% %%% 1/2 for double counting (see Dalfovo Rev. Mod. Phys 71 1999, Eq. 37),
% %%% but the calculation of chemical potential does not.
%
% %%% Kinetic Energy %%%%%%%%%%%%%%%%%%
% % psi_hat = fftn( ifftshift(psi) ); % Use this to speed up, at expense of additional RAM usage
% E_kin_x = 1/2 * sum(sum(sum(  abs( ifftn( 1i*kx.*fftn( ifftshift(psi) ) ) ).^2 )))*dV;
% E_kin_y = 1/2 * sum(sum(sum(  abs( ifftn( 1i*ky.*fftn( ifftshift(psi) ) ) ).^2 )))*dV;
% E_kin_z = 1/2 * sum(sum(sum(  abs( ifftn( 1i*kz.*fftn( ifftshift(psi) ) ) ).^2 )))*dV;
% E_kin = E_kin_x + E_kin_y + E_kin_z;
%
% %%% External Potential Energy %%%%%%%
% %psi_sq = abs(psi).^2;  % Use this to speed up, at expense of additional RAM usage
% E_pot = sum(sum(sum( V_ext.*abs(psi).^2 )))*dV;
%
% %%% Mean Field Interaction Energy %%%
% E_int = G/2 * sum(sum(sum( abs(psi).^4 )))*dV;
%
% %%% Total Energy %%%%%%%%%%%%%%%%%%%%
% E_tot = E_kin + E_pot + E_int;
% E = [E_kin_x ; E_kin_y ; E_kin_z ; E_kin ; E_pot ; E_int ; E_tot];
%
% %%% Chemical Potential %%%%%%%%%%%%%%
% mu = E_kin + E_pot + 2*E_int;
%
% end



function [x,y,z,kx,ky,kz,Tx,Ty,Tz,...
    Nx,Ny,Nz,x0_ind,y0_ind,z0_ind,psi_k] = ...
    expand_grids(DirToExpand, expand_fac, FFT_prime_lengths, dt, Nx, Ny, Nz, dx, dy, dz, psi_k)

if DirToExpand(1) % Expand the grid in x-direction
    Nx = expand_fac*Nx;
    Nx = FFT_prime_lengths( find(FFT_prime_lengths>Nx,1,'first') );
end

if DirToExpand(2) % Expand the grid in y-direction
    Ny = expand_fac*Ny;
    Ny = FFT_prime_lengths( find(FFT_prime_lengths>Ny,1,'first') );
end

if DirToExpand(3) % Expand the grid in z-direction
    Nz = expand_fac*Nz;
    Nz = FFT_prime_lengths( find(FFT_prime_lengths>Nz,1,'first') );
end

% Pad the wavefunction with vacuum (zeroes) around the edges
psi_k = padarray(psi_k, [Ny-size(psi_k,1) Nx-size(psi_k,2) Nz-size(psi_k,3)]/2, 0, 'both');

% Recreate spatial vectors
[x, kx] = grid_vectors(Nx, dx);
[y, ky] = grid_vectors(Ny, dy);
[z, kz] = grid_vectors(Nz, dz);
[~,x0_ind] = min(abs(x));
[~,y0_ind] = min(abs(y));
[~,z0_ind] = min(abs(z));

% Permute variables to enable use of Matlab's implicit expansion
x = permute(x, [1 2 3]); % Creates a [1 x Nx x 1] vector
y = permute(y, [2 1 3]); % Creates a [Ny x 1 x 1] vector
z = permute(z, [3 1 2]); % Creates a [1 x 1 x Nz] vector

% Recreate k-vectors
kx = permute(ifftshift(kx), [1 2 3]);
ky = permute(ifftshift(ky), [2 1 3]);
kz = permute(ifftshift(kz), [3 1 2]);
Tx = exp( -1i*dt/2 * kx.^2);
Ty = exp( -1i*dt/2 * ky.^2);
Tz = exp( -1i*dt/2 * kz.^2);

end

function n_ThomasFermi_3D = ThomasFermi_3D_density(X, Y, Z, omega_x, omega_y, omega_z, N, c)
%%% Calculate 3D Thomas-Fermi limit (3D parabola)

mu_3D = (15*sqrt(2)/(32*pi)*N*c.g_int3D*c.mRb87^(3/2)*omega_x*omega_y*omega_z)^(2/5);
R_x = sqrt( 2*mu_3D/(c.mRb87*omega_x^2) );
R_y = sqrt( 2*mu_3D/(c.mRb87*omega_y^2) );
R_z = sqrt( 2*mu_3D/(c.mRb87*omega_z^2) );

n_ThomasFermi_3D = mu_3D/c.g_int3D * max( (1 - X.^2/R_x^2 - Y.^2/R_y^2 - Z.^2/R_z^2) , 0 );

end

function [MSK_vec_x, MSK_vec_y, MSK_vec_z] = create_mask_vectors(x,y,z,Nx,Ny,Nz,bnd_frac)

[~,bndind_x] = min( abs(x - bnd_frac*x(Nx)) ); % Find the index to begin the masking
x_bound = x(bndind_x);                         % Find the grid coordinate to begin masking
[~,bndind_y] = min( abs(y - bnd_frac*y(Ny)) );
y_bound = y(bndind_y);
[~,bndind_z] = min( abs(z - bnd_frac*z(Nz)) );
z_bound = z(bndind_z);

dbnd_xp = x(Nx) - x_bound;
dbnd_xm = x(1)  + x_bound;
dbnd_yp = y(Ny) - y_bound;
dbnd_ym = y(1)  + y_bound;
dbnd_zp = z(Nz) - z_bound;
dbnd_zm = z(1)  + z_bound;

MSK_vec_x = ones(1,Nx,1);
MSK_vec_y = ones(Ny,1,1);
MSK_vec_z = ones(1,1,Nz);
MSK_vec_x(x >  x_bound) = ( cos( ( x(x>+x_bound) - x_bound )/dbnd_xp * pi/2 ) ).^(1/8);
MSK_vec_x(x < -x_bound) = ( cos( ( x(x<-x_bound) + x_bound )/dbnd_xm * pi/2 ) ).^(1/8);
MSK_vec_y(y >  y_bound) = ( cos( ( y(y>+y_bound) - y_bound )/dbnd_yp * pi/2 ) ).^(1/8);
MSK_vec_y(y < -y_bound) = ( cos( ( y(y<-y_bound) + y_bound )/dbnd_ym * pi/2 ) ).^(1/8);
MSK_vec_z(z >  z_bound) = ( cos( ( z(z>+z_bound) - z_bound )/dbnd_zp * pi/2 ) ).^(1/8);
MSK_vec_z(z < -z_bound) = ( cos( ( z(z<-z_bound) + z_bound )/dbnd_zm * pi/2 ) ).^(1/8);

MSK_vec_x(Nx) = 0;
MSK_vec_x(1)  = 0;
MSK_vec_y(Ny) = 0;
MSK_vec_y(1)  = 0;
MSK_vec_z(Nz) = 0;
MSK_vec_z(1)  = 0;

end


function [E, mu] = calculate_energy_and_mu(psi, dV, kx, ky, kz, V_ext, G, INT_psi)
%%% Calculate total energy of wavefunction (sum of kinetic, potential and
%%% interaction energies). Obtained using the expectation value of the
%%% Hamiltonian operator. Note, interaction energy includes a factor of
%%% 1/2 for double counting (see Dalfovo Rev. Mod. Phys 71 1999, Eq. 37),
%%% but the calculation of chemical potential does not.

%%% Kinetic Energy %%%%%%%%%%%%%%%%%%
% psi_hat = fftn( ifftshift(psi) ); % Use this to speed up, at expense of additional RAM usage
E_kin_x = 1/2 * sum(sum(sum(  abs( ifftn( 1i*kx.*fftn( ifftshift(psi) ) ) ).^2 )))*dV / INT_psi ;
E_kin_y = 1/2 * sum(sum(sum(  abs( ifftn( 1i*ky.*fftn( ifftshift(psi) ) ) ).^2 )))*dV / INT_psi ;
E_kin_z = 1/2 * sum(sum(sum(  abs( ifftn( 1i*kz.*fftn( ifftshift(psi) ) ) ).^2 )))*dV / INT_psi ;
E_kin = E_kin_x + E_kin_y + E_kin_z;

%%% External Potential Energy %%%%%%%
%psi_sq = abs(psi).^2;  % Use this to speed up, at expense of additional RAM usage
E_pot = sum(sum(sum( V_ext.*abs(psi).^2 )))*dV / INT_psi ;

%%% Mean Field Interaction Energy %%%
E_int = G/2 * sum(sum(sum( abs(psi).^4 )))*dV / INT_psi ;

%%% Total Energy %%%%%%%%%%%%%%%%%%%%
E_tot = E_kin + E_pot + E_int;
E = [E_kin_x ; E_kin_y ; E_kin_z ; E_kin ; E_pot ; E_int ; E_tot];

%%% Chemical Potential %%%%%%%%%%%%%%
mu = E_kin + E_pot + 2*E_int;

end