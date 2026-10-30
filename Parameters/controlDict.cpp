deltaT 1.2e-08;
startFrom latestTime;
startTime 0;
endTime 0.3;
// Full-field snapshots: 10 acoustic sampling intervals, in seconds.
writeInterval 1.52587890625e-4;
// Target uniform acoustic grid: 1/65536 s. p/U are saved every solver step.
acousticWriteInterval 1.52587890625e-5;
// Adaptive steps cannot exceed the target acoustic sample interval.
maxDeltaT $acousticWriteInterval;
purgeWrite 10;
adjustTimeStep yes;
// Adaptive growth remains limited by maxCo, the startup cap and maxDeltaT.
deltaTFactor 1.5;
maxCo 5;

impermeableEnabled no;
permeableEnabled no;
