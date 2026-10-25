deltaT 1.2e-08;
startFrom latestTime;
startTime 0;
endTime 0.3;
maxDeltaT 1e-5;
// Full-field snapshots: 10 acoustic sampling intervals, in seconds.
writeInterval 1.52587890625e-4;
// Acoustic surface p/U output: 1/65536 s, matching the measurement rate.
acousticWriteInterval 1.52587890625e-5;
purgeWrite 10;
adjustTimeStep yes;
// Escape output-alignment timestep plateaus while retaining Co and dt caps.
deltaTFactor 1.5;
maxCo 5;

impermeableEnabled no;
permeableEnabled no;
