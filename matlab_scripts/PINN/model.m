function out = model
% model.m  — упрощённая версия экспортированного COMSOL-скрипта (исправлено: dif1 input2 + physics selections)
import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

%% Parameters
model.param.set('rho_plla', '1240[kg/m^3]', '"Density PLLA"');
model.param.set('E_plla_not_complex', '3.5e9[Pa]');
model.param.set('E_plla', 'E_plla_not_complex*(1 + i*mech_loss)', '"Young''s modulus"');
model.param.set('nu_plla', '0.33', '"Poisson coefficient"');
model.param.set('mu_plla', 'E_plla/(2*(1+nu_plla))', 'Lame parameters');
model.param.set('lambda_plla', 'E_plla*nu_plla/((1+nu_plla)*(1-2*nu_plla))', 'Lame parameters');
model.param.set('c_p_plla', 'sqrt((lambda_plla+2*mu_plla)/rho_plla)', '"longitudinal wave velocity"');
model.param.set('c_s_plla', 'sqrt(mu_plla/rho_plla)', '"transverse wave velocity"');
model.param.set('r_cylinder', '10[um]', 'Radius of Cylinder');
model.param.set('h_cylinder', '5[um]', 'Height of Cylinder');
model.param.set('r_int_cylinder', '2*r_cylinder*coeff_modes_check', 'Radius of the sphere for integration');
model.param.set('eps_r_plla', '2.5', 'relative permittivity');
model.param.set('eps0', '8.8541878128e-12[F/m]', 'vacuum permittivity');
model.param.set('epsilon_plla', 'eps0*eps_r_plla*(1 - i * tan_delta)');
model.param.set('d14_plla', '-10e-12[C/N]', 'shear piezo coefficient');
model.param.set('d33_plla', '0.21e-15[C/N]');
model.param.set('mech_loss', '0.01', 'Mechanical loss');
model.param.set('tan_delta', '0.01', 'Dielectric loss');
model.param.set('s_diag', '1/E_plla', 'strain-charge diagonal (11, 22, 33)');
model.param.set('s_off_diag', '-nu_plla/E_plla', 'strain-charge off-diagonal elements s_12');
model.param.set('s_shear', '1/mu_plla', 's_44, s_55, s_66');
model.param.set('c_diag', 'lambda_plla + 2*mu_plla', 'Matrix Strain c_11, c_22, c_33');
model.param.set('e14_plla', 'd14_plla*mu_plla', 'Matrix relations e_14');
model.param.set('r_pml', 'r_int_cylinder*1.5', 'PML');
model.param.set('coeff_modes_check', '1');
model.param.label('Parameters_plla');
model.param.create('par2');
model.param('par2').label('Parameters_common');

model.param('par2').set('c_0', 'c_w', 'Speed of sound in air');
model.param('par2').set('rho_0', 'rho_w', 'Density of air');
model.param('par2').set('p_0', '5 [kPa]', 'Intensity of the incident wave');
model.param('par2').set('f', '30 [MHz]', 'Frequency');
model.param('par2').set('f_start', '1 [MHz]', 'Starting frequency');
model.param('par2').set('f_stop', '160 [MHz]', 'Stop frequency');
model.param('par2').set('rho_w', '1000 [kg/m^3]', 'Density of water');
model.param('par2').set('c_w', '1480 [m/s]', 'Speed of sound in water');
model.param('par2').set('eps_r_w', '80', 'Epsilon relativity water');
model.param('par2').set('lambda_f', 'c_w/(6*f)');
model.param('par2').set('V_0', '-70[mV]');
model.param('par2').set('theta', 'pi/2');
model.param('par2').set('t_src', '1[um]', 'effective "thickness" or length of the source region');
model.param('par2').set('m_modes_max', 'round(r_cylinder * 6 * f_stop / c_w)');
model.param('par2').set('m_modes', '0');
model.param('par2').set('tol', '0.01');
model.param('par2').set('m_centr', '0');

%% Component / Geometry
model.component.create('comp1', true);
model.component('comp1').geom.create('geom1', 3);
model.component('comp1').geom('geom1').geomRep('cadps');
model.component('comp1').geom('geom1').designBooleans(false);

% Cylinder and spheres
model.component('comp1').geom('geom1').create('cyl1', 'Cylinder');
model.component('comp1').geom('geom1').feature('cyl1').set('r', 'r_cylinder');
model.component('comp1').geom('geom1').feature('cyl1').set('h', 'h_cylinder');
model.component('comp1').geom('geom1').feature('cyl1').set('pos', {'0' '0' '-h_cylinder/2'});
model.component('comp1').geom('geom1').run('cyl1');

model.component('comp1').geom('geom1').create('sph1', 'Sphere');
model.component('comp1').geom('geom1').feature('sph1').set('r', 'r_int_cylinder');
model.component('comp1').geom('geom1').run('sph1');

model.component('comp1').geom('geom1').create('sph2', 'Sphere');
model.component('comp1').geom('geom1').feature('sph2').set('r', 'r_pml');
model.component('comp1').geom('geom1').feature('sph2').label('PML');
model.component('comp1').geom('geom1').run('sph2');

% Difference features
model.component('comp1').geom('geom1').create('dif1', 'Difference');
model.component('comp1').geom('geom1').feature.duplicate('dif2', 'dif1');

% Set dif1 inputs: sph2 - sph1
model.component('comp1').geom('geom1').feature('dif1').set('keepsubtract', true);
model.component('comp1').geom('geom1').feature('dif1').selection('input').set({'sph2'});
model.component('comp1').geom('geom1').feature('dif1').selection('input2').set({'sph1'});
model.component('comp1').geom('geom1').run('dif1');

% Configure dif2: sph1 - cyl1
model.component('comp1').geom('geom1').feature('dif2').selection('input').set({'sph1'});
model.component('comp1').geom('geom1').feature('dif2').selection('input2').set({'cyl1'});
model.component('comp1').geom('geom1').feature('dif2').set('keepsubtract', true);
model.component('comp1').geom('geom1').run('dif2');

model.component('comp1').geom('geom1').run;

%% Selections
model.component('comp1').selection.create('sel1', 'Explicit');
model.component('comp1').selection('sel1').set([1]);
model.component('comp1').selection('sel1').label('PML');

model.component('comp1').selection.duplicate('sel2', 'sel1');
model.component('comp1').selection('sel2').label('Integration_selection');
model.component('comp1').selection('sel2').set([2]);

model.component('comp1').selection.create('sel3', 'Explicit');
model.component('comp1').selection('sel3').label('Cylinder_selection');
model.component('comp1').selection('sel3').set([3]);

model.component('comp1').selection.create('sel4', 'Explicit');
model.component('comp1').selection('sel4').geom(2);
model.component('comp1').selection('sel4').set([9 10 11 12 17 22]);
model.component('comp1').selection('sel4').label('cyl_bound_all');

%% Physics
model.component('comp1').physics.create('solid', 'SolidMechanics', 'geom1');
model.component('comp1').physics.create('acpr', 'PressureAcoustics', 'geom1');
model.component('comp1').physics.create('es', 'Electrostatics', 'geom1');

model.component('comp1').physics('solid').create('bndl1', 'BoundaryLoad', 2);
model.component('comp1').physics('solid').create('pzm1', 'PiezoelectricMaterialModel', 3);
model.component('comp1').physics('solid').feature.move('pzm1', 3);
model.component('comp1').physics('solid').feature('pzm1').set('MixedFormulation', 'none');
model.component('comp1').physics('solid').feature('pzm1').selection.named('sel3');

% COMSOL 6.3 - model.component('comp1').physics('solid').feature('bndl1').set('forceType', 'FollowerPressure');
% COMSOL 6.0:
model.component('comp1').physics('solid').feature('bndl1').set('LoadType', 'FollowerPressure');
model.component('comp1').physics('solid').feature('bndl1').set('FollowerPressure', 'acpr.p_s');
%model.component('comp1').physics('solid').feature('bndl1').set('pressure', 'acpr.p_s');
model.component('comp1').physics('solid').feature('bndl1').selection.named('sel4');

model.component('comp1').physics('acpr').create('bpf1', 'BackgroundPressureField', 3);
model.component('comp1').physics('acpr').feature('bpf1').set('dir', [0 0 -1]);
model.component('comp1').physics('acpr').feature('bpf1').set('c', 'c_0');   % using parameter c_0 (== c_w)
model.component('comp1').physics('acpr').feature('bpf1').set('pamp', 'p_0');
% apply to sets: sel2, sel3, sel1
model.component('comp1').physics('acpr').feature('bpf1').selection.named('sel2');
%model.component('comp1').physics('acpr').feature('bpf1').selection.named('sel3');
%model.component('comp1').physics('acpr').feature('bpf1').selection.named('sel1');

model.component('comp1').physics('es').create('ccnp1', 'ChargeConservationPiezo', 3);
model.component('comp1').physics('es').feature('ccnp1').selection.named('sel3');
model.component('comp1').physics('es').create('gnd1', 'Ground', 2);
model.component('comp1').physics('es').feature('gnd1').selection.set([11]);

%% Materials
model.component('comp1').material.create('mat1', 'Common');
model.component('comp1').material('mat1').label('Water');
model.component('comp1').material('mat1').selection.set([1 2]);
model.component('comp1').material('mat1').propertyGroup('def').set('density', {'rho_w'});
model.component('comp1').material('mat1').propertyGroup('def').set('soundspeed', {'c_w'});

model.component('comp1').material.create('mat2', 'Common');
model.component('comp1').material('mat2').label('PLLA');
model.component('comp1').material('mat2').selection.named('sel3');
model.component('comp1').material('mat2').propertyGroup('def').set('density', {'rho_plla'});
% COMSOL 6.3 - model.component('comp1').material('mat2').propertyGroup.create('StressCharge', 'StressCharge', 'Stress-charge_form');

pg = model.component('comp1').material('mat2').propertyGroup().create('StressCharge', 'Stress-charge form');

pg.set('cE', {'c_diag' 'lambda_plla' 'c_diag' 'lambda_plla' 'lambda_plla' 'c_diag' ...
    '0' '0' '0' 'mu_plla' '0' '0' '0' '0' 'mu_plla' '0' '0' '0' '0' '0' 'mu_plla'});

pg.set('eES', {'0' '0' '0' '0' '0' '0' '0' '0' '0' 'e14_plla' ...
    '0' '0' '0' '0' '0' '0' '0' '0'});

pg.set('epsilonrS', {'eps_r_plla'});

% model.component('comp1').material('mat2').propertyGroup('StressCharge').set('cE', {'c_diag' 'lambda_plla' 'c_diag' 'lambda_plla' 'lambda_plla' 'c_diag' ...
%     '0' '0' '0' 'mu_plla' '0' '0' '0' '0' 'mu_plla' '0' '0' '0' '0' '0' 'mu_plla'});
% model.component('comp1').material('mat2').propertyGroup('StressCharge').set('eES', {'0' '0' '0' '0' '0' '0' '0' '0' '0' 'e14_plla' ...
%     '0' '0' '0' '0' '0' '0' '0' '0'});
% model.component('comp1').material('mat2').propertyGroup('StressCharge').set('epsilonrS', {'eps_r_plla'});

%% Multiphysics / Couplings
model.component('comp1').multiphysics.create('pze1', 'PiezoelectricEffect', 3);
model.component('comp1').multiphysics.create('asb1', 'AcousticStructureBoundary', 2);
model.component('comp1').multiphysics('asb1').selection.named('sel4');

%% PML coord system
model.component('comp1').coordSystem.create('pml1', 'PML');
model.component('comp1').coordSystem('pml1').selection.named('sel1');
model.component('comp1').coordSystem('pml1').set('ScalingType', 'Spherical');

%% Mesh
model.component('comp1').mesh.create('mesh1');
%COMSOL 6.3 - model.component('comp1').mesh('mesh1').contribute('geom/detail', true);

model.component('comp1').mesh('mesh1').create('ftet1', 'FreeTet');
model.component('comp1').mesh('mesh1').feature('ftet1').create('size1', 'Size');
model.component('comp1').mesh('mesh1').feature('ftet1').selection.geom('geom1', 3);
model.component('comp1').mesh('mesh1').feature('ftet1').selection.named('sel3');
model.component('comp1').mesh('mesh1').feature('ftet1').label('Cylinder');

model.component('comp1').mesh('mesh1').feature.duplicate('ftet2', 'ftet1');
model.component('comp1').mesh('mesh1').feature('ftet2').label('Integration');
model.component('comp1').mesh('mesh1').feature('ftet2').selection.named('sel2');
model.component('comp1').mesh('mesh1').feature.duplicate('ftet3', 'ftet2');
model.component('comp1').mesh('mesh1').feature('ftet3').label('PML');
model.component('comp1').mesh('mesh1').feature('ftet3').selection.named('sel1');

% Default mesh size (run_batch изменяет при необходимости)
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('custom', true);
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmin', '0.9E-6');
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hmax',  '2.4E-6');
model.component('comp1').mesh('mesh1').feature('ftet1').feature('size1').set('hgrad', 1.3);
model.component('comp1').mesh('mesh1').run;

%% Study
model.study.create('std1');
model.study('std1').create('freq', 'Frequency');
model.study('std1').feature('freq').setSolveFor('/physics/solid', true);
model.study('std1').feature('freq').setSolveFor('/physics/acpr', true);
model.study('std1').feature('freq').setSolveFor('/physics/es', true);
model.study('std1').feature('freq').setSolveFor('/multiphysics/pze1', true);
model.study('std1').feature('freq').setSolveFor('/multiphysics/asb1', true);
model.study('std1').feature('freq').set('plist', 'range(f_start, (f_stop - f_start)/2, f_stop)');
model.study('std1').createAutoSequences('all');

%% Coupling operators (intops)
model.component('comp1').cpl.create('intop1', 'Integration');
model.component('comp1').cpl('intop1').set('axisym', true);
model.component('comp1').cpl('intop1').label('Integration_for_voltage');
model.component('comp1').cpl('intop1').set('opname', 'intop_top');
model.component('comp1').cpl('intop1').selection.geom('geom1', 2);
model.component('comp1').cpl('intop1').selection.named('sel4');
model.component('comp1').cpl('intop1').selection.set([12]);

model.component('comp1').cpl.create('intop2', 'Integration');
model.component('comp1').cpl('intop2').set('axisym', true);
model.component('comp1').cpl('intop2').set('opname', 'intop');
model.component('comp1').cpl('intop2').selection.geom('geom1', 2);
model.component('comp1').cpl('intop2').selection.set([5 6 7 8 15 16 19 20]);

%% Variables
model.component('comp1').variable.create('var1');
model.component('comp1').variable('var1').set('vs_x', 'd(acpr.p_s, x)/(-i*rho_0*acpr.omega)', 'Scattered field speed, x component');
model.component('comp1').variable('var1').set('vs_y', 'd(acpr.p_s, y)/(-i*rho_0*acpr.omega)', 'Scattered field speed, y component');
model.component('comp1').variable('var1').set('vs_z', 'd(acpr.p_s, z)/(-i*rho_0*acpr.omega)', 'Scattered field speed, z component');
model.component('comp1').variable('var1').set('s_x', 'real( conj(acpr.p_s) * vs_x )/2', 'Poynting vector of scattered field, x component');
model.component('comp1').variable('var1').set('s_y', 'real( conj(acpr.p_s) * vs_y)/2', 'Poynting vector of scattered field, y component');
model.component('comp1').variable('var1').set('s_z', 'real( conj(acpr.p_s) * vs_z )/2', 'Poynting vector of scattered field, z component');
model.component('comp1').variable('var1').set('I_inc', 'abs(p_0)^2 / (2*c_0*rho_0)', 'Intensity of the incident plane wave');
model.component('comp1').variable('var1').set('P_ang', '(s_x * x + s_y * y + s_z * z) * sqrt(x^2 + y^2 + z^2)', 'Angular Dependance of scattered power');
model.component('comp1').variable('var1').set('P_sc', 'intop(s_x*nx + s_y*ny +s_z*nz)', 'Total scattered power');
model.component('comp1').variable('var1').set('D_sc', 'P_ang/P_sc', 'Directivity');
model.component('comp1').variable('var1').set('sigma_sc', 'P_sc / I_inc', 'Scattering cross-section');
model.component('comp1').variable('var1').set('sigma_geom', 'pi*r_cylinder^2', 'Geomertical cross-section');
model.component('comp1').variable('var1').set('Q_sc', 'sigma_sc / sigma_geom', 'Scattering efficiency');

%% EvalGlobal fallback (gev1/tbl1)
try model.result.numerical.create('gev1', 'EvalGlobal'); catch; end
model.result.numerical('gev1').setIndex('expr', 'intop_top(V)/intop_top(1)', 0);
try model.result.table.create('tbl1', 'Table'); catch; end
model.result.numerical('gev1').set('table', 'tbl1');

%% --- ВАЖНО: назначаем домены физикам (было пропущено ранее) ---
% Эти строки необходимы, чтобы PressureAcoustics видел материал (soundspeed/c_w) в доменах 1 и 2
model.component('comp1').physics('solid').selection.set([3]);
model.component('comp1').physics('acpr').selection.set([1 2]);
model.component('comp1').physics('es').selection.set([3]);

out = model;
end