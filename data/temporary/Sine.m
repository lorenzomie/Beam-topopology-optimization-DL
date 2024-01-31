%%% SINE WAVE GENERATION %%%
cycles = 5;
time = 1e-3;
sf = 150e3; % Sine frequency [Hz]
disc = 400;
 
t = linspace(0,cycles*1/sf,disc);
sinewave = sin(2*pi*sf*t); % Continuous sine wave generation
w = hanning(length(sinewave)); % Hanning windowing
sinewave = [sinewave 0 0];
t = [t time time];
w = [w' 0 0];
 
figure
plot(t, sinewave.*w)
xlabel('time [s]')
ylabel('amplitude [N]')
Abaqus_input_tuples = [t' (sinewave.*w)'];
disp(Abaqus_input_tuples);

% File Saving
fid = fopen('Abaqus_input.txt', 'w');
fprintf(fid, '%.8f, %.8f\n', Abaqus_input_tuples');
fclose(fid);

