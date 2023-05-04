% Physical constants

function c = physical_constants()

c.hbar = 1.055e-34;        % Reduced Planck constant
c.kB = 1.38e-23;           % Boltzman constant
c.a0 = 5.291772e-11;       % Bohr radius
c.amu = 1.660539e-27;      % Atomic mass unit
c.zeta_3 = 1.202057;       % Riemann Zeta function (3)
c.zeta_2 = 1.644934;       % Riemann Zeta function (2)
c.aRb87 = 98.98*c.a0;      % Scattering length, Rb87 S=1 triplet state (van Kempen, Phys. Rev. Lett. 88 (2002),p. 093201)
c.mRb87 = 86.909181*c.amu; % Mass of Rb87 (Daniel Steck, Rb87 D Line Data, https://steck.us/alkalidata/rubidium87numbers.1.6.pdf)
c.g = 9.81;                % Gravitational acceleration
c.g_int3D = 4*pi*c.hbar^2*c.aRb87/c.mRb87; % 3D interaction coupling constant