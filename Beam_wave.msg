1

   Abaqus 2020                                  Date 04-Feb-2024   Time 21:11:50
   For use by Supplied by Team-SolidSQUAD under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     D Y N A M I C   A N A L Y S I S - I M P L I C I T  I N T E G R A T I O N


                                                                                          

          DYNAMIC APPLICATION PACKED SCHEME IS                 TRANSIENT FIDELITY  
          HILBER-HUGHES-TAYLOR (HHT) TIME INTEGRATOR IS USED

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                1.000E-06
          AND A TOTAL TIME PERIOD OF                           1.000E-02
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.000E-07
          THE MAXIMUM TIME INCREMENT ALLOWED IS                1.000E-04
          SCALE FACTOR ON TIME AVERAGE FORCE FOR HALF-
           -INCREMENT RESIDUAL COMPUTATIONS IS                 1.000E+03

          HILBER-HUGHES-TAYLOR METHOD PARAMETERS:  ALPHA   IS  -5.000000E-02
                                                   BETA    IS   0.275625    
                                                   GAMMA   IS   0.550000    
          SINGULAR MASS TREATMENT TYPE IS                     ERROR
          CONSERVATIVE TIME INCREMENTATION IS USED
          STEP AMPLITUDE IS                                    STEP

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE FORCE                               1.000E-02
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR MOMENT   
     CRITERION FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR ROTATION CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE MOMENT                              1.000E-02
     AVERAGE MOMENT    IS TIME AVERAGE MOMENT   
     ALTERNATE CRIT. FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO MOMENT    RELATIVE TO TIME AVRG. MOMENT        1.000E-05
     CRITERION FOR RESIDUAL MOMENT    WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR ROTATION CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL MOMENT    FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO MOMENT    REL. TO TIME AVRG. MAX. MOMENT       1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CONSECUTIVE INCS MEETING TIME INTEG. TOL. FOR TIME INC. INCREASE          3
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     CUT-BACK FACTOR WHEN TIME INTEGRATION ACCURACY TOL. EXCEEDED     0.8500    
     MAX. RATIO OF TIME INTEG. MEASURE TO TOL. FOR TIME INC. INCREASE 0.7500    
     INCREASE FACTOR FOR TIME INCREMENT WHEN POSSIBLE                 0.8000    
     MIN. TIME INC. INCREASE FACTOR IN LINEAR TRANSIENT PROBLEMS      0.9500    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                 1000000

     VELOCITY-BASED PARABOLIC EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH      2.00    

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.824E-04   TIME AVG. FORCE       3.824E-04
 LARGEST RESIDUAL FORCE             5.421E-20   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.850E-08   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.850E-08   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.202E-06   TIME AVG. MOMENT      6.202E-06
 LARGEST RESIDUAL MOMENT            1.626E-19   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.550E-10   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.550E-10   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   4.195E-03 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -2.200E-04 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  1.000E-04
 STEP TIME COMPLETED       1.000E-06,  TOTAL TIME COMPLETED        1.000E-06


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.994E-03   TIME AVG. FORCE       2.188E-03
 LARGEST RESIDUAL FORCE            -1.355E-19   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.269E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        7.161E-08   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.105E-04   TIME AVG. MOMENT      3.584E-04
 LARGEST RESIDUAL MOMENT           -1.735E-18   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.578E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.115E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.388E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -9.008E-04 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  2.000E-04
 STEP TIME COMPLETED       2.000E-06,  TOTAL TIME COMPLETED        2.000E-06


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.673E-03   TIME AVG. FORCE       2.683E-03
 LARGEST RESIDUAL FORCE             1.735E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.087E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.386E-08   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.150E-03   TIME AVG. MOMENT      9.556E-04
 LARGEST RESIDUAL MOMENT           -4.337E-19   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      6.873E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.766E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.707E-03 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -5.204E-04 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    1.250E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-06,  FRACTION OF STEP COMPLETED  3.000E-04
 STEP TIME COMPLETED       3.000E-06,  TOTAL TIME COMPLETED        3.000E-06


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.250E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.011E-02   TIME AVG. FORCE       4.541E-03
 LARGEST RESIDUAL FORCE             4.163E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.009E-07   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -5.862E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.539E-03   TIME AVG. MOMENT      1.351E-03
 LARGEST RESIDUAL MOMENT           -5.204E-18   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.914E-08   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.160E-09   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -7.455E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   6.466E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    1.562E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.250E-06,  FRACTION OF STEP COMPLETED  4.250E-04
 STEP TIME COMPLETED       4.250E-06,  TOTAL TIME COMPLETED        4.250E-06


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.562E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.524E-02   TIME AVG. FORCE       6.680E-03
 LARGEST RESIDUAL FORCE            -5.551E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -1.951E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -7.881E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.629E-03   TIME AVG. MOMENT      2.407E-03
 LARGEST RESIDUAL MOMENT            1.640E-17   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.303E-08   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.468E-08   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.765E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.050E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    1.953E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.562E-06,  FRACTION OF STEP COMPLETED  5.812E-04
 STEP TIME COMPLETED       5.812E-06,  TOTAL TIME COMPLETED        5.812E-06


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.953E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.666E-02   TIME AVG. FORCE       1.001E-02
 LARGEST RESIDUAL FORCE            -1.110E-16   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -1.280E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        5.306E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     9.671E-03   TIME AVG. MOMENT      3.618E-03
 LARGEST RESIDUAL MOMENT            5.551E-17   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.305E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.133E-08   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.330     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -5.682E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    2.441E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.953E-06,  FRACTION OF STEP COMPLETED  7.766E-04
 STEP TIME COMPLETED       7.766E-06,  TOTAL TIME COMPLETED        7.766E-06


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.441E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.613E-02   TIME AVG. FORCE       1.088E-02
 LARGEST RESIDUAL FORCE            -5.551E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.239E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.535E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.599E-02   TIME AVG. MOMENT      5.385E-03
 LARGEST RESIDUAL MOMENT           -6.939E-17   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.306E-07   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.913E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.282     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   6.803E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    3.052E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.441E-06,  FRACTION OF STEP COMPLETED  1.021E-03
 STEP TIME COMPLETED       1.021E-05,  TOTAL TIME COMPLETED        1.021E-05


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.052E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.736E-03   TIME AVG. FORCE       1.074E-02
 LARGEST RESIDUAL FORCE            -1.318E-16   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.419E-06   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.058E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.724E-02   TIME AVG. MOMENT      6.867E-03
 LARGEST RESIDUAL MOMENT            9.714E-17   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.924E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.889E-07   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   6.487E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   9.271E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    3.815E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.052E-06,  FRACTION OF STEP COMPLETED  1.326E-03
 STEP TIME COMPLETED       1.326E-05,  TOTAL TIME COMPLETED        1.326E-05


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.815E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.530E-02   TIME AVG. FORCE       1.236E-02
 LARGEST RESIDUAL FORCE             4.163E-16   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -1.356E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -7.685E-06   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.137E-02   TIME AVG. MOMENT      9.590E-03
 LARGEST RESIDUAL MOMENT            3.691E-16   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.444E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.286E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.112     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.126     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    4.768E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.815E-06,  FRACTION OF STEP COMPLETED  1.707E-03
 STEP TIME COMPLETED       1.707E-05,  TOTAL TIME COMPLETED        1.707E-05


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.768E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.113E-02   TIME AVG. FORCE       1.424E-02
 LARGEST RESIDUAL FORCE             5.829E-16   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -7.235E-06   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.855E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.151E-02   TIME AVG. MOMENT      1.278E-02
 LARGEST RESIDUAL MOMENT            6.661E-16   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.016E-06   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.150E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.829     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.439     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    5.960E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.768E-06,  FRACTION OF STEP COMPLETED  2.184E-03
 STEP TIME COMPLETED       2.184E-05,  TOTAL TIME COMPLETED        2.184E-05


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.960E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.363E-02   TIME AVG. FORCE       1.691E-02
 LARGEST RESIDUAL FORCE            -2.220E-16   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.263E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.835E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.239E-02   TIME AVG. MOMENT      1.820E-02
 LARGEST RESIDUAL MOMENT           -2.220E-15   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.623E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.744E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.983     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.912     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    7.451E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.960E-06,  FRACTION OF STEP COMPLETED  2.780E-03
 STEP TIME COMPLETED       2.780E-05,  TOTAL TIME COMPLETED        2.780E-05


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.451E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.145E-02   TIME AVG. FORCE       1.812E-02
 LARGEST RESIDUAL FORCE             2.816E-16   AT NODE        499   DOF  1
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.809E-05   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -5.600E-05   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.172       TIME AVG. MOMENT      3.100E-02
 LARGEST RESIDUAL MOMENT            1.920E-15   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      7.411E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.396E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.11     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.19     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    9.313E-06

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.451E-06,  FRACTION OF STEP COMPLETED  3.525E-03
 STEP TIME COMPLETED       3.525E-05,  TOTAL TIME COMPLETED        3.525E-05


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.313E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.653E-02   TIME AVG. FORCE       1.954E-02
 LARGEST RESIDUAL FORCE            -1.707E-15   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.588E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        6.165E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.184       TIME AVG. MOMENT      4.277E-02
 LARGEST RESIDUAL MOMENT            4.559E-15   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      5.729E-06   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.432E-05   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.969     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.73     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    1.164E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.313E-06,  FRACTION OF STEP COMPLETED  4.457E-03
 STEP TIME COMPLETED       4.457E-05,  TOTAL TIME COMPLETED        4.457E-05


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.164E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.464E-02   TIME AVG. FORCE       2.061E-02
 LARGEST RESIDUAL FORCE            -2.331E-15   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.752E-05   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.035E-04   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.138       TIME AVG. MOMENT      4.960E-02
 LARGEST RESIDUAL MOMENT           -5.523E-15   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.641E-06   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.950E-05   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.04     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    2.52     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    1.455E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.164E-05,  FRACTION OF STEP COMPLETED  5.621E-03
 STEP TIME COMPLETED       5.621E-05,  TOTAL TIME COMPLETED        5.621E-05


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.455E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.936E-02   TIME AVG. FORCE       2.120E-02
 LARGEST RESIDUAL FORCE            -2.459E-15   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.125E-05   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.286E-04   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.128       TIME AVG. MOMENT      5.482E-02
 LARGEST RESIDUAL MOMENT            8.993E-15   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.243E-06   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.790E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.27     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -3.83     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    1.819E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.455E-05,  FRACTION OF STEP COMPLETED  7.076E-03
 STEP TIME COMPLETED       7.076E-05,  TOTAL TIME COMPLETED        7.076E-05


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.819E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.523E-02   TIME AVG. FORCE       2.145E-02
 LARGEST RESIDUAL FORCE            -2.026E-15   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.233E-05   AT NODE         99   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.239E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.129       TIME AVG. MOMENT      5.945E-02
 LARGEST RESIDUAL MOMENT            1.205E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.585E-06   AT NODE        491   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.601E-05   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.42     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    5.03     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    2.274E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.819E-05,  FRACTION OF STEP COMPLETED  8.895E-03
 STEP TIME COMPLETED       8.895E-05,  TOTAL TIME COMPLETED        8.895E-05


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.274E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.508E-02   TIME AVG. FORCE       2.166E-02
 LARGEST RESIDUAL FORCE            -2.616E-15   AT NODE        492   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.162E-05   AT NODE        491   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.441E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.121       TIME AVG. MOMENT      6.308E-02
 LARGEST RESIDUAL MOMENT           -1.027E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.200E-06   AT NODE        487   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.934E-05   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -4.57     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -6.17     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    2.842E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.274E-05,  FRACTION OF STEP COMPLETED  1.117E-02
 STEP TIME COMPLETED       1.117E-04,  TOTAL TIME COMPLETED        1.117E-04


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.842E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.202E-02   TIME AVG. FORCE       2.168E-02
 LARGEST RESIDUAL FORCE             2.678E-15   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.046E-05   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.480E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.129       TIME AVG. MOMENT      6.674E-02
 LARGEST RESIDUAL MOMENT            1.787E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.484E-06   AT NODE        484   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.885E-05   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    7.32     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -8.42     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    3.553E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.842E-05,  FRACTION OF STEP COMPLETED  1.401E-02
 STEP TIME COMPLETED       1.401E-04,  TOTAL TIME COMPLETED        1.401E-04


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.553E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.904E-02   TIME AVG. FORCE       2.154E-02
 LARGEST RESIDUAL FORCE             6.826E-15   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.942E-05   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.451E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.123       TIME AVG. MOMENT      6.969E-02
 LARGEST RESIDUAL MOMENT           -7.383E-15   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.397E-06   AT NODE        479   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.991E-05   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -10.8     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    11.9     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    4.441E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.553E-05,  FRACTION OF STEP COMPLETED  1.756E-02
 STEP TIME COMPLETED       1.756E-04,  TOTAL TIME COMPLETED        1.756E-04


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.441E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.700E-02   TIME AVG. FORCE       2.132E-02
 LARGEST RESIDUAL FORCE            -7.081E-15   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.527E-05   AT NODE        469   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.861E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.114       TIME AVG. MOMENT      7.190E-02
 LARGEST RESIDUAL MOMENT            1.166E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.433E-06   AT NODE        474   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.027E-05   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    15.3     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -16.0     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    5.551E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.441E-05,  FRACTION OF STEP COMPLETED  2.200E-02
 STEP TIME COMPLETED       2.200E-04,  TOTAL TIME COMPLETED        2.200E-04


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.551E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.462E-02   TIME AVG. FORCE       2.100E-02
 LARGEST RESIDUAL FORCE             4.198E-15   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.556E-05   AT NODE        463   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.297E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.106       TIME AVG. MOMENT      7.351E-02
 LARGEST RESIDUAL MOMENT           -2.498E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.447E-06   AT NODE        469   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.012E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -20.9     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    21.0     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  2.756E-02
 STEP TIME COMPLETED       2.756E-04,  TOTAL TIME COMPLETED        2.756E-04


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.551E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.296E-02   TIME AVG. FORCE       2.063E-02
 LARGEST RESIDUAL FORCE            -4.569E-15   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.639E-05   AT NODE        457   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.858E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.101       TIME AVG. MOMENT      7.476E-02
 LARGEST RESIDUAL MOMENT           -1.363E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.903E-06   AT NODE        463   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.747E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    18.2     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -18.6     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  3.311E-02
 STEP TIME COMPLETED       3.311E-04,  TOTAL TIME COMPLETED        3.311E-04


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.551E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.138E-02   TIME AVG. FORCE       2.023E-02
 LARGEST RESIDUAL FORCE            -4.538E-15   AT NODE        456   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -7.385E-05   AT NODE        451   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.727E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.970E-02   TIME AVG. MOMENT      7.541E-02
 LARGEST RESIDUAL MOMENT            1.792E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.837E-06   AT NODE        445   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.666E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -15.6     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    18.1     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  3.866E-02
 STEP TIME COMPLETED       3.866E-04,  TOTAL TIME COMPLETED        3.866E-04


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.551E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.054E-02   TIME AVG. FORCE       1.983E-02
 LARGEST RESIDUAL FORCE             5.714E-15   AT NODE        484   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -7.288E-05   AT NODE         87   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.536E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.521E-02   TIME AVG. MOMENT      7.581E-02
 LARGEST RESIDUAL MOMENT           -1.177E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.737E-06   AT NODE        439   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.434E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    13.2     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -17.3     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  4.421E-02
 STEP TIME COMPLETED       4.421E-04,  TOTAL TIME COMPLETED        4.421E-04


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.551E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.583E-03   TIME AVG. FORCE       1.942E-02
 LARGEST RESIDUAL FORCE            -7.983E-15   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.689E-05   AT NODE        440   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.345E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.358E-02   TIME AVG. MOMENT      7.613E-02
 LARGEST RESIDUAL MOMENT            1.221E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.634E-06   AT NODE        433   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.103E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -11.0     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    16.4     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  4.976E-02
 STEP TIME COMPLETED       4.976E-04,  TOTAL TIME COMPLETED        4.976E-04


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.551E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      8.662E-03   TIME AVG. FORCE       1.900E-02
 LARGEST RESIDUAL FORCE             9.819E-15   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.771E-05   AT NODE         84   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.142E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.773E-02   TIME AVG. MOMENT      7.619E-02
 LARGEST RESIDUAL MOMENT            1.761E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.614E-06   AT NODE        426   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.820E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    9.06     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -15.3     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    6.939E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  5.531E-02
 STEP TIME COMPLETED       5.531E-04,  TOTAL TIME COMPLETED        5.531E-04


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.939E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.875E-03   TIME AVG. FORCE       1.859E-02
 LARGEST RESIDUAL FORCE            -4.677E-15   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.867E-05   AT NODE        428   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.339E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.194E-02   TIME AVG. MOMENT      7.603E-02
 LARGEST RESIDUAL MOMENT            2.338E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -4.059E-06   AT NODE        419   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.649E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    11.6     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    21.4     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    8.439E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.939E-05,  FRACTION OF STEP COMPLETED  6.225E-02
 STEP TIME COMPLETED       6.225E-04,  TOTAL TIME COMPLETED        6.225E-04


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.439E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.093E-03   TIME AVG. FORCE       1.818E-02
 LARGEST RESIDUAL FORCE            -7.686E-15   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.341E-05   AT NODE        405   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.464E-04   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.730E-02   TIME AVG. MOMENT      7.572E-02
 LARGEST RESIDUAL MOMENT           -1.509E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -4.421E-06   AT NODE        413   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.946E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -15.1     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -27.8     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.439E-05,  FRACTION OF STEP COMPLETED  7.069E-02
 STEP TIME COMPLETED       7.069E-04,  TOTAL TIME COMPLETED        7.069E-04


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.439E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.497E-03   TIME AVG. FORCE       1.778E-02
 LARGEST RESIDUAL FORCE             8.760E-15   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.084E-05   AT NODE        397   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.263E-04   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.723E-02   TIME AVG. MOMENT      7.543E-02
 LARGEST RESIDUAL MOMENT            8.243E-15   AT NODE        475   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.834E-06   AT NODE         77   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.128E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    13.4     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    24.5     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.439E-05,  FRACTION OF STEP COMPLETED  7.913E-02
 STEP TIME COMPLETED       7.913E-04,  TOTAL TIME COMPLETED        7.913E-04


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.439E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.926E-03   TIME AVG. FORCE       1.738E-02
 LARGEST RESIDUAL FORCE             1.073E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.361E-05   AT NODE        389   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.196E-04   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.463E-02   TIME AVG. MOMENT      7.507E-02
 LARGEST RESIDUAL MOMENT           -8.125E-15   AT NODE        485   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.377E-06   AT NODE         75   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.719E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -11.8     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -21.4     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.439E-05,  FRACTION OF STEP COMPLETED  8.757E-02
 STEP TIME COMPLETED       8.757E-04,  TOTAL TIME COMPLETED        8.757E-04


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.439E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.409E-03   TIME AVG. FORCE       1.700E-02
 LARGEST RESIDUAL FORCE             6.828E-15   AT NODE        479   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.482E-05   AT NODE         71   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.126E-04   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.120E-02   TIME AVG. MOMENT      7.462E-02
 LARGEST RESIDUAL MOMENT           -1.067E-14   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.261E-06   AT NODE        372   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.361E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    10.3     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    18.6     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.439E-05,  FRACTION OF STEP COMPLETED  9.601E-02
 STEP TIME COMPLETED       9.601E-04,  TOTAL TIME COMPLETED        9.601E-04


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.439E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.983E-03   TIME AVG. FORCE       1.662E-02
 LARGEST RESIDUAL FORCE             9.976E-15   AT NODE        484   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.741E-05   AT NODE         69   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.075E-04   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.120E-02   TIME AVG. MOMENT      7.420E-02
 LARGEST RESIDUAL MOMENT            1.496E-14   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.422E-06   AT NODE        365   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.053E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -8.92     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -16.0     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    1.000E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.439E-05,  FRACTION OF STEP COMPLETED  0.104    
 STEP TIME COMPLETED       1.044E-03,  TOTAL TIME COMPLETED        1.044E-03


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.588E-03   TIME AVG. FORCE       1.626E-02
 LARGEST RESIDUAL FORCE             7.388E-15   AT NODE         98   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         7.587E-05   AT NODE        366   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.320E-04   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.966E-02   TIME AVG. MOMENT      7.376E-02
 LARGEST RESIDUAL MOMENT            7.654E-15   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.605E-06   AT NODE         65   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.292E-05   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    10.6     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    20.9     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.114    
 STEP TIME COMPLETED       1.144E-03,  TOTAL TIME COMPLETED        1.144E-03


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.193E-03   TIME AVG. FORCE       1.590E-02
 LARGEST RESIDUAL FORCE            -7.211E-15   AT NODE        480   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.863E-05   AT NODE        358   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.119E-04   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.478E-02   TIME AVG. MOMENT      7.320E-02
 LARGEST RESIDUAL MOMENT            9.732E-15   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.366E-06   AT NODE        349   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.991E-05   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -9.00     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -19.8     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.124    
 STEP TIME COMPLETED       1.244E-03,  TOTAL TIME COMPLETED        1.244E-03


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.915E-03   TIME AVG. FORCE       1.556E-02
 LARGEST RESIDUAL FORCE            -8.428E-15   AT NODE        467   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.652E-05   AT NODE        350   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.087E-04   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.194E-02   TIME AVG. MOMENT      7.259E-02
 LARGEST RESIDUAL MOMENT           -7.604E-15   AT NODE         96   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.147E-06   AT NODE        340   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.883E-05   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    7.53     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    18.6     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.134    
 STEP TIME COMPLETED       1.344E-03,  TOTAL TIME COMPLETED        1.344E-03


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.608E-03   TIME AVG. FORCE       1.523E-02
 LARGEST RESIDUAL FORCE             1.041E-14   AT NODE        478   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.641E-05   AT NODE        321   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.043E-04   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.831E-02   TIME AVG. MOMENT      7.192E-02
 LARGEST RESIDUAL MOMENT            9.909E-15   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.220E-06   AT NODE        330   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.754E-05   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    6.46     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -17.5     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.144    
 STEP TIME COMPLETED       1.444E-03,  TOTAL TIME COMPLETED        1.444E-03


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.332E-03   TIME AVG. FORCE       1.491E-02
 LARGEST RESIDUAL FORCE             1.086E-14   AT NODE         95   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.943E-05   AT NODE        313   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.966E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.748E-02   TIME AVG. MOMENT      7.126E-02
 LARGEST RESIDUAL MOMENT           -1.074E-14   AT NODE        480   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.129E-06   AT NODE        322   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.610E-05   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -5.97     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    16.3     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.154    
 STEP TIME COMPLETED       1.544E-03,  TOTAL TIME COMPLETED        1.544E-03


  INCREMENT    38 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.122E-03   TIME AVG. FORCE       1.460E-02
 LARGEST RESIDUAL FORCE            -1.374E-14   AT NODE        479   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.812E-05   AT NODE        305   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        9.149E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.751E-02   TIME AVG. MOMENT      7.063E-02
 LARGEST RESIDUAL MOMENT           -1.150E-14   AT NODE        480   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.968E-06   AT NODE        314   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.475E-05   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    5.49     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -15.2     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.164    
 STEP TIME COMPLETED       1.644E-03,  TOTAL TIME COMPLETED        1.644E-03


  INCREMENT    39 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.959E-03   TIME AVG. FORCE       1.430E-02
 LARGEST RESIDUAL FORCE             1.156E-14   AT NODE        480   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.535E-05   AT NODE        297   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -8.239E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.821E-02   TIME AVG. MOMENT      7.006E-02
 LARGEST RESIDUAL MOMENT            8.094E-15   AT NODE        484   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.843E-06   AT NODE        287   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.365E-05   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -5.02     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    14.1     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.174    
 STEP TIME COMPLETED       1.744E-03,  TOTAL TIME COMPLETED        1.744E-03


  INCREMENT    40 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.798E-03   TIME AVG. FORCE       1.401E-02
 LARGEST RESIDUAL FORCE            -9.073E-15   AT NODE        472   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.806E-05   AT NODE        289   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        8.146E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.750E-02   TIME AVG. MOMENT      6.949E-02
 LARGEST RESIDUAL MOMENT           -8.344E-15   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.752E-06   AT NODE        278   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.234E-05   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    4.56     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -13.0     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.184    
 STEP TIME COMPLETED       1.844E-03,  TOTAL TIME COMPLETED        1.844E-03


  INCREMENT    41 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.642E-03   TIME AVG. FORCE       1.373E-02
 LARGEST RESIDUAL FORCE            -1.160E-14   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.318E-05   AT NODE        281   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -7.881E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.865E-02   TIME AVG. MOMENT      6.899E-02
 LARGEST RESIDUAL MOMENT           -1.066E-14   AT NODE        476   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.673E-06   AT NODE         43   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.107E-05   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -4.13     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    11.9     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.194    
 STEP TIME COMPLETED       1.944E-03,  TOTAL TIME COMPLETED        1.944E-03


  INCREMENT    42 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.516E-03   TIME AVG. FORCE       1.347E-02
 LARGEST RESIDUAL FORCE            -1.558E-14   AT NODE        477   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.913E-05   AT NODE        250   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        7.304E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.738E-02   TIME AVG. MOMENT      6.847E-02
 LARGEST RESIDUAL MOMENT           -8.349E-15   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.556E-06   AT NODE         41   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.999E-06   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -3.97     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -10.9     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.204    
 STEP TIME COMPLETED       2.044E-03,  TOTAL TIME COMPLETED        2.044E-03


  INCREMENT    43 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.442E-03   TIME AVG. FORCE       1.321E-02
 LARGEST RESIDUAL FORCE            -1.856E-14   AT NODE        476   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.008E-05   AT NODE         36   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -6.719E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.646E-02   TIME AVG. MOMENT      6.796E-02
 LARGEST RESIDUAL MOMENT           -1.083E-14   AT NODE        477   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.398E-06   AT NODE        253   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.960E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    4.08     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    9.99     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.214    
 STEP TIME COMPLETED       2.144E-03,  TOTAL TIME COMPLETED        2.144E-03


  INCREMENT    44 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.366E-03   TIME AVG. FORCE       1.296E-02
 LARGEST RESIDUAL FORCE            -1.248E-14   AT NODE        482   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.080E-05   AT NODE         34   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        6.308E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.515E-02   TIME AVG. MOMENT      6.744E-02
 LARGEST RESIDUAL MOMENT           -1.377E-14   AT NODE        483   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.305E-06   AT NODE        222   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.080E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -4.12     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -9.10     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.224    
 STEP TIME COMPLETED       2.244E-03,  TOTAL TIME COMPLETED        2.244E-03


  INCREMENT    45 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.253E-03   TIME AVG. FORCE       1.272E-02
 LARGEST RESIDUAL FORCE            -1.736E-14   AT NODE        482   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.198E-05   AT NODE         32   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -5.905E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.294E-02   TIME AVG. MOMENT      6.690E-02
 LARGEST RESIDUAL MOMENT           -1.002E-14   AT NODE        471   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.516E-06   AT NODE         29   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.174E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    4.10     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    8.26     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.234    
 STEP TIME COMPLETED       2.344E-03,  TOTAL TIME COMPLETED        2.344E-03


  INCREMENT    46 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.203E-03   TIME AVG. FORCE       1.250E-02
 LARGEST RESIDUAL FORCE            -1.425E-14   AT NODE        484   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.053E-05   AT NODE         30   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        5.599E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.492E-02   TIME AVG. MOMENT      6.642E-02
 LARGEST RESIDUAL MOMENT           -1.407E-14   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.611E-06   AT NODE        206   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.264E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -4.04     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -7.47     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.244    
 STEP TIME COMPLETED       2.444E-03,  TOTAL TIME COMPLETED        2.444E-03


  INCREMENT    47 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.157E-03   TIME AVG. FORCE       1.228E-02
 LARGEST RESIDUAL FORCE             1.822E-14   AT NODE         94   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.223E-05   AT NODE         28   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -5.050E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.603E-02   TIME AVG. MOMENT      6.599E-02
 LARGEST RESIDUAL MOMENT            1.001E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.652E-06   AT NODE        198   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.537E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    3.94     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    6.74     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.254    
 STEP TIME COMPLETED       2.544E-03,  TOTAL TIME COMPLETED        2.544E-03


  INCREMENT    48 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.067E-03   TIME AVG. FORCE       1.206E-02
 LARGEST RESIDUAL FORCE            -2.126E-14   AT NODE         95   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.061E-05   AT NODE         26   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.312E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.568E-02   TIME AVG. MOMENT      6.556E-02
 LARGEST RESIDUAL MOMENT           -1.317E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.579E-06   AT NODE        190   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.956E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -3.81     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -6.06     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.264    
 STEP TIME COMPLETED       2.644E-03,  TOTAL TIME COMPLETED        2.644E-03


  INCREMENT    49 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.000E-03   TIME AVG. FORCE       1.186E-02
 LARGEST RESIDUAL FORCE            -2.350E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.481E-05   AT NODE        194   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.030E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.571E-02   TIME AVG. MOMENT      6.516E-02
 LARGEST RESIDUAL MOMENT            1.514E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.277E-06   AT NODE        183   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.281E-06   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    3.65     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    5.43     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.274    
 STEP TIME COMPLETED       2.744E-03,  TOTAL TIME COMPLETED        2.744E-03


  INCREMENT    50 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.077E-03   TIME AVG. FORCE       1.166E-02
 LARGEST RESIDUAL FORCE             1.958E-14   AT NODE        468   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.584E-05   AT NODE        163   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.924E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.625E-02   TIME AVG. MOMENT      6.478E-02
 LARGEST RESIDUAL MOMENT            1.245E-14   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.145E-06   AT NODE        151   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.061E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -3.48     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -4.86     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.284    
 STEP TIME COMPLETED       2.844E-03,  TOTAL TIME COMPLETED        2.844E-03


  INCREMENT    51 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.870E-03   TIME AVG. FORCE       1.147E-02
 LARGEST RESIDUAL FORCE             1.577E-14   AT NODE        459   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.635E-05   AT NODE        154   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.399E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.249E-02   TIME AVG. MOMENT      6.434E-02
 LARGEST RESIDUAL MOMENT           -1.228E-14   AT NODE        486   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.206E-06   AT NODE        143   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.619E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    3.31     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -4.39     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.294    
 STEP TIME COMPLETED       2.944E-03,  TOTAL TIME COMPLETED        2.944E-03


  INCREMENT    52 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.670E-03   TIME AVG. FORCE       1.128E-02
 LARGEST RESIDUAL FORCE             2.042E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.253E-05   AT NODE        123   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.846E-05   AT NODE        149   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.197E-02   TIME AVG. MOMENT      6.391E-02
 LARGEST RESIDUAL MOMENT           -1.353E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.940E-06   AT NODE        135   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.140E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -3.12     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    4.05     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.304    
 STEP TIME COMPLETED       3.044E-03,  TOTAL TIME COMPLETED        3.044E-03


  INCREMENT    53 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.996E-03   TIME AVG. FORCE       1.111E-02
 LARGEST RESIDUAL FORCE             1.754E-14   AT NODE         95   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.153E-05   AT NODE         11   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.291E-05   AT NODE          6   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.635E-02   TIME AVG. MOMENT      6.358E-02
 LARGEST RESIDUAL MOMENT            1.564E-14   AT NODE        475   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.141E-06   AT NODE        130   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.041E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.93     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -3.73     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.314    
 STEP TIME COMPLETED       3.144E-03,  TOTAL TIME COMPLETED        3.144E-03


  INCREMENT    54 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.556E-03   TIME AVG. FORCE       1.093E-02
 LARGEST RESIDUAL FORCE             1.580E-14   AT NODE        455   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.827E-05   AT NODE          7   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.521E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.217E-02   TIME AVG. MOMENT      6.318E-02
 LARGEST RESIDUAL MOMENT            1.387E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.469E-06   AT NODE        137   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.858E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -2.74     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    3.42     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.324    
 STEP TIME COMPLETED       3.244E-03,  TOTAL TIME COMPLETED        3.244E-03


  INCREMENT    55 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.518E-03   TIME AVG. FORCE       1.076E-02
 LARGEST RESIDUAL FORCE            -2.139E-14   AT NODE        482   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -9.112E-05   AT NODE        123   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.451E-05   AT NODE          6   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.943E-02   TIME AVG. MOMENT      6.275E-02
 LARGEST RESIDUAL MOMENT            1.167E-14   AT NODE         96   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.829E-06   AT NODE        133   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.562E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.55     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -3.13     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.334    
 STEP TIME COMPLETED       3.344E-03,  TOTAL TIME COMPLETED        3.344E-03


  INCREMENT    56 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.496E-03   TIME AVG. FORCE       1.059E-02
 LARGEST RESIDUAL FORCE             3.538E-14   AT NODE        481   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.067E-05   AT NODE        128   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.169E-05   AT NODE        120   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.957E-02   TIME AVG. MOMENT      6.234E-02
 LARGEST RESIDUAL MOMENT           -1.951E-14   AT NODE        480   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.219E-06   AT NODE        144   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.370E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -2.37     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    2.85     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.344    
 STEP TIME COMPLETED       3.444E-03,  TOTAL TIME COMPLETED        3.444E-03


  INCREMENT    57 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.587E-03   TIME AVG. FORCE       1.043E-02
 LARGEST RESIDUAL FORCE             2.075E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.801E-05   AT NODE        136   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.278E-05   AT NODE        137   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.342E-02   TIME AVG. MOMENT      6.201E-02
 LARGEST RESIDUAL MOMENT            1.461E-14   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.452E-06   AT NODE        146   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.172E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.19     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -2.59     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.354    
 STEP TIME COMPLETED       3.544E-03,  TOTAL TIME COMPLETED        3.544E-03


  INCREMENT    58 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.470E-03   TIME AVG. FORCE       1.028E-02
 LARGEST RESIDUAL FORCE            -1.825E-14   AT NODE         91   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.270E-05   AT NODE        193   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.464E-05   AT NODE        119   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.564E-02   TIME AVG. MOMENT      6.172E-02
 LARGEST RESIDUAL MOMENT           -1.445E-14   AT NODE         92   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.869E-06   AT NODE        181   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.015E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -2.02     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    2.35     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.364    
 STEP TIME COMPLETED       3.644E-03,  TOTAL TIME COMPLETED        3.644E-03


  INCREMENT    59 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.442E-03   TIME AVG. FORCE       1.013E-02
 LARGEST RESIDUAL FORCE            -2.479E-14   AT NODE         96   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.233E-05   AT NODE        133   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.074E-05   AT NODE        151   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.146E-02   TIME AVG. MOMENT      6.138E-02
 LARGEST RESIDUAL MOMENT            1.964E-14   AT NODE        481   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.417E-06   AT NODE        143   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.896E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.85     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -2.12     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.374    
 STEP TIME COMPLETED       3.744E-03,  TOTAL TIME COMPLETED        3.744E-03


  INCREMENT    60 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.303E-03   TIME AVG. FORCE       9.983E-03
 LARGEST RESIDUAL FORCE             1.450E-14   AT NODE        484   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.975E-05   AT NODE        136   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.715E-05   AT NODE        134   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.633E-02   TIME AVG. MOMENT      6.096E-02
 LARGEST RESIDUAL MOMENT            1.393E-14   AT NODE        480   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.157E-06   AT NODE        152   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.707E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.69     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.91     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.384    
 STEP TIME COMPLETED       3.844E-03,  TOTAL TIME COMPLETED        3.844E-03


  INCREMENT    61 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.372E-03   TIME AVG. FORCE       9.842E-03
 LARGEST RESIDUAL FORCE            -1.566E-14   AT NODE        475   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.491E-05   AT NODE        170   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.934E-05   AT NODE        166   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.708E-02   TIME AVG. MOMENT      6.057E-02
 LARGEST RESIDUAL MOMENT           -1.110E-14   AT NODE         91   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.472E-06   AT NODE        157   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.775E-06   AT NODE        157   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.54     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.71     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.394    
 STEP TIME COMPLETED       3.944E-03,  TOTAL TIME COMPLETED        3.944E-03


  INCREMENT    62 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.387E-03   TIME AVG. FORCE       9.706E-03
 LARGEST RESIDUAL FORCE             2.442E-14   AT NODE         96   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.092E-05   AT NODE        178   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.961E-05   AT NODE        197   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.082E-02   TIME AVG. MOMENT      6.025E-02
 LARGEST RESIDUAL MOMENT           -1.880E-14   AT NODE        476   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.461E-06   AT NODE         23   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.599E-06   AT NODE        188   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.40     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.53     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.404    
 STEP TIME COMPLETED       4.044E-03,  TOTAL TIME COMPLETED        4.044E-03


  INCREMENT    63 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.482E-03   TIME AVG. FORCE       9.575E-03
 LARGEST RESIDUAL FORCE            -2.677E-14   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.391E-05   AT NODE        185   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.194E-05   AT NODE        132   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.419E-02   TIME AVG. MOMENT      6.000E-02
 LARGEST RESIDUAL MOMENT            1.903E-14   AT NODE        483   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.481E-06   AT NODE        198   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.415E-06   AT NODE        125   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.27     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.36     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.414    
 STEP TIME COMPLETED       4.144E-03,  TOTAL TIME COMPLETED        4.144E-03


  INCREMENT    64 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.533E-03   TIME AVG. FORCE       9.450E-03
 LARGEST RESIDUAL FORCE             2.278E-14   AT NODE        476   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.260E-05   AT NODE        218   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.761E-05   AT NODE        192   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.473E-02   TIME AVG. MOMENT      5.976E-02
 LARGEST RESIDUAL MOMENT           -1.964E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.213E-06   AT NODE        204   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.622E-06   AT NODE        201   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.15     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.21     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.424    
 STEP TIME COMPLETED       4.244E-03,  TOTAL TIME COMPLETED        4.244E-03


  INCREMENT    65 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.593E-03   TIME AVG. FORCE       9.329E-03
 LARGEST RESIDUAL FORCE            -2.797E-14   AT NODE        471   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.534E-05   AT NODE        225   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.450E-05   AT NODE        221   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.539E-02   TIME AVG. MOMENT      5.954E-02
 LARGEST RESIDUAL MOMENT           -1.775E-14   AT NODE         94   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.126E-06   AT NODE        212   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.356E-06   AT NODE        212   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.04     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.07     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.434    
 STEP TIME COMPLETED       4.344E-03,  TOTAL TIME COMPLETED        4.344E-03


  INCREMENT    66 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.479E-03   TIME AVG. FORCE       9.210E-03
 LARGEST RESIDUAL FORCE            -2.818E-14   AT NODE        477   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.387E-05   AT NODE        233   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.896E-05   AT NODE        231   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.658E-02   TIME AVG. MOMENT      5.934E-02
 LARGEST RESIDUAL MOMENT            2.073E-14   AT NODE        476   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.793E-06   AT NODE        245   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.143E-06   AT NODE        241   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.930     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.960     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.444    
 STEP TIME COMPLETED       4.444E-03,  TOTAL TIME COMPLETED        4.444E-03


  INCREMENT    67 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.400E-03   TIME AVG. FORCE       9.093E-03
 LARGEST RESIDUAL FORCE            -2.847E-14   AT NODE         93   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.822E-05   AT NODE        241   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.947E-05   AT NODE        261   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.413E-02   TIME AVG. MOMENT      5.911E-02
 LARGEST RESIDUAL MOMENT           -1.866E-14   AT NODE        470   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.951E-06   AT NODE         39   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.092E-06   AT NODE        209   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.832     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.889     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.454    
 STEP TIME COMPLETED       4.544E-03,  TOTAL TIME COMPLETED        4.544E-03


  INCREMENT    68 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.453E-03   TIME AVG. FORCE       8.981E-03
 LARGEST RESIDUAL FORCE             2.996E-14   AT NODE        467   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.652E-05   AT NODE        275   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.972E-05   AT NODE        245   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.993E-02   TIME AVG. MOMENT      5.883E-02
 LARGEST RESIDUAL MOMENT            2.292E-14   AT NODE         95   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.939E-06   AT NODE         41   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.084E-06   AT NODE        236   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.742     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.820     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.464    
 STEP TIME COMPLETED       4.644E-03,  TOTAL TIME COMPLETED        4.644E-03


  INCREMENT    69 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.318E-03   TIME AVG. FORCE       8.870E-03
 LARGEST RESIDUAL FORCE             2.390E-14   AT NODE        475   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.148E-05   AT NODE        282   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.931E-05   AT NODE        255   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.780E-02   TIME AVG. MOMENT      5.853E-02
 LARGEST RESIDUAL MOMENT            1.745E-14   AT NODE        481   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.831E-06   AT NODE        269   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.094E-06   AT NODE        264   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.659     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.754     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.474    
 STEP TIME COMPLETED       4.744E-03,  TOTAL TIME COMPLETED        4.744E-03


  INCREMENT    70 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.324E-03   TIME AVG. FORCE       8.762E-03
 LARGEST RESIDUAL FORCE            -3.426E-14   AT NODE        465   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.499E-05   AT NODE        290   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.911E-05   AT NODE        284   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.806E-02   TIME AVG. MOMENT      5.824E-02
 LARGEST RESIDUAL MOMENT           -1.710E-14   AT NODE        473   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.795E-06   AT NODE        277   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.084E-06   AT NODE        275   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.584     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.691     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.484    
 STEP TIME COMPLETED       4.844E-03,  TOTAL TIME COMPLETED        4.844E-03


  INCREMENT    71 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.339E-03   TIME AVG. FORCE       8.658E-03
 LARGEST RESIDUAL FORCE             3.107E-14   AT NODE         96   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.688E-05   AT NODE         50   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.096E-05   AT NODE         49   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.948E-02   TIME AVG. MOMENT      5.797E-02
 LARGEST RESIDUAL MOMENT           -1.758E-14   AT NODE        483   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.848E-06   AT NODE         53   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.802E-07   AT NODE         51   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.515     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.631     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.494    
 STEP TIME COMPLETED       4.944E-03,  TOTAL TIME COMPLETED        4.944E-03


  INCREMENT    72 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.364E-03   TIME AVG. FORCE       8.556E-03
 LARGEST RESIDUAL FORCE            -2.640E-14   AT NODE        485   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.610E-05   AT NODE        305   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.087E-05   AT NODE        302   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.938E-02   TIME AVG. MOMENT      5.771E-02
 LARGEST RESIDUAL MOMENT            1.885E-14   AT NODE        484   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.832E-06   AT NODE        317   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.086E-06   AT NODE        311   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.453     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.575     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.504    
 STEP TIME COMPLETED       5.044E-03,  TOTAL TIME COMPLETED        5.044E-03


  INCREMENT    73 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.465E-03   TIME AVG. FORCE       8.459E-03
 LARGEST RESIDUAL FORCE             3.018E-14   AT NODE        479   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.322E-05   AT NODE        312   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.943E-05   AT NODE        310   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.750E-02   TIME AVG. MOMENT      5.744E-02
 LARGEST RESIDUAL MOMENT           -2.371E-14   AT NODE         92   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.755E-06   AT NODE        324   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.055E-06   AT NODE        320   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.397     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.522     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.514    
 STEP TIME COMPLETED       5.144E-03,  TOTAL TIME COMPLETED        5.144E-03


  INCREMENT    74 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.456E-03   TIME AVG. FORCE       8.364E-03
 LARGEST RESIDUAL FORCE             3.759E-14   AT NODE        479   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.934E-05   AT NODE        320   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.803E-05   AT NODE        318   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.649E-02   TIME AVG. MOMENT      5.715E-02
 LARGEST RESIDUAL MOMENT           -2.987E-14   AT NODE        475   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.740E-06   AT NODE        332   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.828E-07   AT NODE        329   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.346     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.473     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.524    
 STEP TIME COMPLETED       5.244E-03,  TOTAL TIME COMPLETED        5.244E-03


  INCREMENT    75 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.473E-03   TIME AVG. FORCE       8.273E-03
 LARGEST RESIDUAL FORCE            -3.677E-14   AT NODE        460   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.180E-05   AT NODE        326   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.633E-05   AT NODE        325   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.933E-02   TIME AVG. MOMENT      5.692E-02
 LARGEST RESIDUAL MOMENT            1.901E-14   AT NODE        465   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.654E-06   AT NODE        340   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     9.327E-07   AT NODE        315   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.300     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.427     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.534    
 STEP TIME COMPLETED       5.344E-03,  TOTAL TIME COMPLETED        5.344E-03


  INCREMENT    76 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.581E-03   TIME AVG. FORCE       8.185E-03
 LARGEST RESIDUAL FORCE             2.836E-14   AT NODE        468   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.070E-05   AT NODE         66   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.763E-05   AT NODE        357   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.256E-02   TIME AVG. MOMENT      5.673E-02
 LARGEST RESIDUAL MOMENT            1.909E-14   AT NODE        470   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.649E-06   AT NODE        349   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.174E-07   AT NODE        346   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.259     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.384     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.544    
 STEP TIME COMPLETED       5.444E-03,  TOTAL TIME COMPLETED        5.444E-03


  INCREMENT    77 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.566E-03   TIME AVG. FORCE       8.099E-03
 LARGEST RESIDUAL FORCE             3.378E-14   AT NODE        480   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.491E-05   AT NODE        369   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.905E-05   AT NODE        364   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.510E-02   TIME AVG. MOMENT      5.658E-02
 LARGEST RESIDUAL MOMENT            2.377E-14   AT NODE        465   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.668E-06   AT NODE        357   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.192E-07   AT NODE        356   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.223     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.345     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.554    
 STEP TIME COMPLETED       5.544E-03,  TOTAL TIME COMPLETED        5.544E-03


  INCREMENT    78 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.524E-03   TIME AVG. FORCE       8.014E-03
 LARGEST RESIDUAL FORCE             3.372E-14   AT NODE        467   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.502E-05   AT NODE        376   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.844E-05   AT NODE        372   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.686E-02   TIME AVG. MOMENT      5.645E-02
 LARGEST RESIDUAL MOMENT           -1.890E-14   AT NODE         90   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.605E-06   AT NODE        364   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.094E-07   AT NODE        363   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.190     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.309     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.564    
 STEP TIME COMPLETED       5.644E-03,  TOTAL TIME COMPLETED        5.644E-03


  INCREMENT    79 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.609E-03   TIME AVG. FORCE       7.933E-03
 LARGEST RESIDUAL FORCE            -2.795E-14   AT NODE         94   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.413E-05   AT NODE        384   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.652E-05   AT NODE        381   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.883E-02   TIME AVG. MOMENT      5.635E-02
 LARGEST RESIDUAL MOMENT            1.932E-14   AT NODE        472   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.453E-06   AT NODE        397   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -7.690E-07   AT NODE        415   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.161     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.276     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.574    
 STEP TIME COMPLETED       5.744E-03,  TOTAL TIME COMPLETED        5.744E-03


  INCREMENT    80 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.638E-03   TIME AVG. FORCE       7.855E-03
 LARGEST RESIDUAL FORCE            -3.825E-14   AT NODE         90   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.290E-05   AT NODE        391   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.700E-05   AT NODE        411   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.873E-02   TIME AVG. MOMENT      5.626E-02
 LARGEST RESIDUAL MOMENT            2.373E-14   AT NODE        472   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.490E-06   AT NODE        405   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.772E-07   AT NODE         76   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.136     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.246     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.584    
 STEP TIME COMPLETED       5.844E-03,  TOTAL TIME COMPLETED        5.844E-03


  INCREMENT    81 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.554E-03   TIME AVG. FORCE       7.777E-03
 LARGEST RESIDUAL FORCE            -3.303E-14   AT NODE        459   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.317E-05   AT NODE        399   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.692E-05   AT NODE        419   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.671E-02   TIME AVG. MOMENT      5.614E-02
 LARGEST RESIDUAL MOMENT            2.180E-14   AT NODE        473   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.575E-06   AT NODE        412   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.341E-07   AT NODE        410   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.113     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.218     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.594    
 STEP TIME COMPLETED       5.944E-03,  TOTAL TIME COMPLETED        5.944E-03


  INCREMENT    82 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.427E-03   TIME AVG. FORCE       7.699E-03
 LARGEST RESIDUAL FORCE             3.152E-14   AT NODE         92   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.645E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.554E-05   AT NODE        427   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.540E-02   TIME AVG. MOMENT      5.601E-02
 LARGEST RESIDUAL MOMENT           -1.968E-14   AT NODE         93   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.551E-06   AT NODE        444   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.325E-07   AT NODE         80   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   9.664E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.193     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.604    
 STEP TIME COMPLETED       6.044E-03,  TOTAL TIME COMPLETED        6.044E-03


  INCREMENT    83 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.485E-03   TIME AVG. FORCE       7.624E-03
 LARGEST RESIDUAL FORCE            -2.358E-14   AT NODE        469   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.087E-05   AT NODE         79   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.039E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.665E-02   TIME AVG. MOMENT      5.590E-02
 LARGEST RESIDUAL MOMENT           -2.443E-14   AT NODE         93   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.466E-06   AT NODE        450   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.090E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -8.380E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.171     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.614    
 STEP TIME COMPLETED       6.144E-03,  TOTAL TIME COMPLETED        6.144E-03


  INCREMENT    84 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.362E-03   TIME AVG. FORCE       7.550E-03
 LARGEST RESIDUAL FORCE            -3.405E-14   AT NODE        478   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.747E-05   AT NODE        421   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.604E-05   AT NODE        444   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.380E-02   TIME AVG. MOMENT      5.575E-02
 LARGEST RESIDUAL MOMENT            2.077E-14   AT NODE        476   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.248E-06   AT NODE        436   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.068E-06   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -7.343E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.150     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.624    
 STEP TIME COMPLETED       6.244E-03,  TOTAL TIME COMPLETED        6.244E-03


  INCREMENT    85 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.304E-04   TIME AVG. FORCE       7.465E-03
 LARGEST RESIDUAL FORCE             3.174E-14   AT NODE        469   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.442E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.430E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.585E-02   TIME AVG. MOMENT      5.552E-02
 LARGEST RESIDUAL MOMENT            1.728E-14   AT NODE         93   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.195E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.439E-07   AT NODE        442   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   6.715E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.132     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.634    
 STEP TIME COMPLETED       6.344E-03,  TOTAL TIME COMPLETED        6.344E-03


  INCREMENT    86 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.242E-03   TIME AVG. FORCE       7.393E-03
 LARGEST RESIDUAL FORCE             2.359E-14   AT NODE        478   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.077E-05   AT NODE        484   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.937E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.044E-02   TIME AVG. MOMENT      5.534E-02
 LARGEST RESIDUAL MOMENT            2.090E-14   AT NODE        468   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.221E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.070E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -6.136E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.115     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.644    
 STEP TIME COMPLETED       6.444E-03,  TOTAL TIME COMPLETED        6.444E-03


  INCREMENT    87 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.230E-03   TIME AVG. FORCE       7.322E-03
 LARGEST RESIDUAL FORCE             3.320E-14   AT NODE        470   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.357E-05   AT NODE        153   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.470E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.130E-02   TIME AVG. MOMENT      5.518E-02
 LARGEST RESIDUAL MOMENT           -2.207E-14   AT NODE        462   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      9.977E-07   AT NODE        411   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.414E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   5.627E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.100     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.654    
 STEP TIME COMPLETED       6.544E-03,  TOTAL TIME COMPLETED        6.544E-03


  INCREMENT    88 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.455E-04   TIME AVG. FORCE       7.249E-03
 LARGEST RESIDUAL FORCE             2.657E-14   AT NODE        477   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         9.741E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.129E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.349E-02   TIME AVG. MOMENT      5.494E-02
 LARGEST RESIDUAL MOMENT           -2.768E-14   AT NODE        467   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.780E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.112E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -5.075E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   8.730E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.664    
 STEP TIME COMPLETED       6.644E-03,  TOTAL TIME COMPLETED        6.644E-03


  INCREMENT    89 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.113E-03   TIME AVG. FORCE       7.180E-03
 LARGEST RESIDUAL FORCE             2.418E-14   AT NODE         94   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.059E-04   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.462E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.430E-02   TIME AVG. MOMENT      5.470E-02
 LARGEST RESIDUAL MOMENT            2.168E-14   AT NODE         96   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.039E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.387E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   4.569E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -7.568E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.674    
 STEP TIME COMPLETED       6.744E-03,  TOTAL TIME COMPLETED        6.744E-03


  INCREMENT    90 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.091E-03   TIME AVG. FORCE       7.113E-03
 LARGEST RESIDUAL FORCE            -3.011E-14   AT NODE        481   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.144E-05   AT NODE        447   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.956E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.706E-02   TIME AVG. MOMENT      5.451E-02
 LARGEST RESIDUAL MOMENT           -2.141E-14   AT NODE        456   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.822E-06   AT NODE        435   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.714E-07   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.146E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -6.695E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.684    
 STEP TIME COMPLETED       6.844E-03,  TOTAL TIME COMPLETED        6.844E-03


  INCREMENT    91 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.191E-03   TIME AVG. FORCE       7.048E-03
 LARGEST RESIDUAL FORCE             2.542E-14   AT NODE        457   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.617E-05   AT NODE        444   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.031E-05   AT NODE        446   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.296E-02   TIME AVG. MOMENT      5.438E-02
 LARGEST RESIDUAL MOMENT           -2.340E-14   AT NODE        463   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.360E-06   AT NODE        431   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.271E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   3.707E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   6.146E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.694    
 STEP TIME COMPLETED       6.944E-03,  TOTAL TIME COMPLETED        6.944E-03


  INCREMENT    92 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.243E-03   TIME AVG. FORCE       6.985E-03
 LARGEST RESIDUAL FORCE            -2.718E-14   AT NODE        458   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.834E-05   AT NODE        384   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.906E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.968E-02   TIME AVG. MOMENT      5.422E-02
 LARGEST RESIDUAL MOMENT            2.530E-14   AT NODE         96   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.476E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.121E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -3.285E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -5.605E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.704    
 STEP TIME COMPLETED       7.044E-03,  TOTAL TIME COMPLETED        7.044E-03


  INCREMENT    93 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.114E-03   TIME AVG. FORCE       6.921E-03
 LARGEST RESIDUAL FORCE             3.764E-14   AT NODE         92   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.164E-05   AT NODE        472   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.797E-05   AT NODE        450   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.708E-02   TIME AVG. MOMENT      5.404E-02
 LARGEST RESIDUAL MOMENT           -2.290E-14   AT NODE         90   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.719E-06   AT NODE        458   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     9.327E-07   AT NODE        441   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   2.924E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   5.119E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.714    
 STEP TIME COMPLETED       7.144E-03,  TOTAL TIME COMPLETED        7.144E-03


  INCREMENT    94 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.117E-03   TIME AVG. FORCE       6.860E-03
 LARGEST RESIDUAL FORCE             3.695E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.890E-05   AT NODE        471   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.330E-05   AT NODE        446   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.833E-02   TIME AVG. MOMENT      5.387E-02
 LARGEST RESIDUAL MOMENT            2.162E-14   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.720E-06   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.410E-07   AT NODE        458   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -2.603E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -4.672E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.724    
 STEP TIME COMPLETED       7.244E-03,  TOTAL TIME COMPLETED        7.244E-03


  INCREMENT    95 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.057E-04   TIME AVG. FORCE       6.792E-03
 LARGEST RESIDUAL FORCE             3.323E-14   AT NODE        470   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.493E-05   AT NODE        408   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.839E-05   AT NODE         79   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.339E-02   TIME AVG. MOMENT      5.376E-02
 LARGEST RESIDUAL MOMENT            1.740E-14   AT NODE        471   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.231E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     9.684E-07   AT NODE        423   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   2.284E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   4.228E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.734    
 STEP TIME COMPLETED       7.344E-03,  TOTAL TIME COMPLETED        7.344E-03


  INCREMENT    96 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.343E-03   TIME AVG. FORCE       6.735E-03
 LARGEST RESIDUAL FORCE            -3.012E-14   AT NODE        460   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.455E-05   AT NODE         76   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.503E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.583E-02   TIME AVG. MOMENT      5.368E-02
 LARGEST RESIDUAL MOMENT            1.571E-14   AT NODE        467   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.622E-06   AT NODE        388   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.897E-07   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.980E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -3.804E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.744    
 STEP TIME COMPLETED       7.444E-03,  TOTAL TIME COMPLETED        7.444E-03


  INCREMENT    97 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.458E-03   TIME AVG. FORCE       6.681E-03
 LARGEST RESIDUAL FORCE            -2.869E-14   AT NODE         90   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.482E-05   AT NODE        396   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.423E-05   AT NODE        398   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.800E-02   TIME AVG. MOMENT      5.362E-02
 LARGEST RESIDUAL MOMENT           -2.368E-14   AT NODE        466   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.554E-06   AT NODE        381   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.952E-07   AT NODE         78   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.787E-02 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   3.433E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.754    
 STEP TIME COMPLETED       7.544E-03,  TOTAL TIME COMPLETED        7.544E-03


  INCREMENT    98 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.432E-03   TIME AVG. FORCE       6.627E-03
 LARGEST RESIDUAL FORCE            -3.229E-14   AT NODE        464   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.943E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.551E-05   AT NODE         67   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.053E-02   TIME AVG. MOMENT      5.359E-02
 LARGEST RESIDUAL MOMENT            2.662E-14   AT NODE        476   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.524E-06   AT NODE         69   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -7.300E-07   AT NODE        375   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.662E-02 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -3.096E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.764    
 STEP TIME COMPLETED       7.644E-03,  TOTAL TIME COMPLETED        7.644E-03


  INCREMENT    99 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.329E-03   TIME AVG. FORCE       6.573E-03
 LARGEST RESIDUAL FORCE            -2.900E-14   AT NODE        474   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.334E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.231E-05   AT NODE        382   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.576E-02   TIME AVG. MOMENT      5.351E-02
 LARGEST RESIDUAL MOMENT            2.844E-14   AT NODE        456   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.255E-06   AT NODE        341   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.241E-07   AT NODE        393   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.549E-02 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -2.855E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.774    
 STEP TIME COMPLETED       7.744E-03,  TOTAL TIME COMPLETED        7.744E-03


  INCREMENT   100 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.223E-03   TIME AVG. FORCE       6.520E-03
 LARGEST RESIDUAL FORCE            -3.065E-14   AT NODE        476   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         7.028E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.302E-05   AT NODE        371   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.997E-02   TIME AVG. MOMENT      5.337E-02
 LARGEST RESIDUAL MOMENT            2.265E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.712E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -7.097E-07   AT NODE         66   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.445E-02 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   2.652E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.784    
 STEP TIME COMPLETED       7.844E-03,  TOTAL TIME COMPLETED        7.844E-03


  INCREMENT   101 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.205E-03   TIME AVG. FORCE       6.467E-03
 LARGEST RESIDUAL FORCE             2.853E-14   AT NODE        481   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         7.584E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.327E-05   AT NODE        318   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.125E-02   TIME AVG. MOMENT      5.325E-02
 LARGEST RESIDUAL MOMENT           -2.620E-14   AT NODE        458   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.095E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.730E-07   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.331E-02 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -2.440E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.794    
 STEP TIME COMPLETED       7.944E-03,  TOTAL TIME COMPLETED        7.944E-03


  INCREMENT   102 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.256E-03   TIME AVG. FORCE       6.416E-03
 LARGEST RESIDUAL FORCE            -3.175E-14   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.906E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.227E-05   AT NODE         53   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.826E-02   TIME AVG. MOMENT      5.320E-02
 LARGEST RESIDUAL MOMENT            2.903E-14   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.315E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.989E-07   AT NODE        346   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.219E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   2.226E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.804    
 STEP TIME COMPLETED       8.044E-03,  TOTAL TIME COMPLETED        8.044E-03


  INCREMENT   103 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.374E-03   TIME AVG. FORCE       6.367E-03
 LARGEST RESIDUAL FORCE            -4.724E-14   AT NODE        491   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.288E-05   AT NODE        423   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.301E-05   AT NODE        324   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.102E-02   TIME AVG. MOMENT      5.318E-02
 LARGEST RESIDUAL MOMENT            2.760E-14   AT NODE        490   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.213E-06   AT NODE         47   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.852E-07   AT NODE        314   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.196E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -2.039E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.814    
 STEP TIME COMPLETED       8.144E-03,  TOTAL TIME COMPLETED        8.144E-03


  INCREMENT   104 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.453E-03   TIME AVG. FORCE       6.320E-03
 LARGEST RESIDUAL FORCE             3.489E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.894E-05   AT NODE        265   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.286E-05   AT NODE        317   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.224E-02   TIME AVG. MOMENT      5.317E-02
 LARGEST RESIDUAL MOMENT           -2.107E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.335E-06   AT NODE        325   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.559E-07   AT NODE        305   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.175E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.872E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.824    
 STEP TIME COMPLETED       8.244E-03,  TOTAL TIME COMPLETED        8.244E-03


  INCREMENT   105 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.482E-03   TIME AVG. FORCE       6.274E-03
 LARGEST RESIDUAL FORCE            -3.992E-14   AT NODE        490   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.504E-05   AT NODE         40   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.356E-05   AT NODE        310   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.138E-02   TIME AVG. MOMENT      5.316E-02
 LARGEST RESIDUAL MOMENT            2.720E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.385E-06   AT NODE        318   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.112E-07   AT NODE        320   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.137E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.709E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.834    
 STEP TIME COMPLETED       8.344E-03,  TOTAL TIME COMPLETED        8.344E-03


  INCREMENT   106 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.479E-03   TIME AVG. FORCE       6.229E-03
 LARGEST RESIDUAL FORCE            -3.409E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.986E-05   AT NODE         50   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.263E-05   AT NODE        301   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.843E-02   TIME AVG. MOMENT      5.311E-02
 LARGEST RESIDUAL MOMENT           -2.743E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.370E-06   AT NODE        311   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.408E-07   AT NODE        290   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.075E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.543E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.844    
 STEP TIME COMPLETED       8.444E-03,  TOTAL TIME COMPLETED        8.444E-03


  INCREMENT   107 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.387E-03   TIME AVG. FORCE       6.184E-03
 LARGEST RESIDUAL FORCE             3.131E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.114E-05   AT NODE        291   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.138E-05   AT NODE         43   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.536E-02   TIME AVG. MOMENT      5.304E-02
 LARGEST RESIDUAL MOMENT            2.441E-14   AT NODE         93   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.213E-06   AT NODE        305   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.347E-07   AT NODE        258   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.015E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.392E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.854    
 STEP TIME COMPLETED       8.544E-03,  TOTAL TIME COMPLETED        8.544E-03


  INCREMENT   108 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.276E-03   TIME AVG. FORCE       6.138E-03
 LARGEST RESIDUAL FORCE            -3.172E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.997E-05   AT NODE        285   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.125E-05   AT NODE        262   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.362E-02   TIME AVG. MOMENT      5.295E-02
 LARGEST RESIDUAL MOMENT            2.178E-14   AT NODE        479   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.213E-06   AT NODE        270   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.541E-07   AT NODE         44   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   9.681E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.263E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.864    
 STEP TIME COMPLETED       8.644E-03,  TOTAL TIME COMPLETED        8.644E-03


  INCREMENT   109 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.270E-03   TIME AVG. FORCE       6.093E-03
 LARGEST RESIDUAL FORCE             4.053E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.829E-05   AT NODE        465   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.123E-05   AT NODE        254   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.155E-02   TIME AVG. MOMENT      5.285E-02
 LARGEST RESIDUAL MOMENT           -2.354E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.180E-06   AT NODE        263   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.321E-07   AT NODE         42   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -9.200E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.144E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.874    
 STEP TIME COMPLETED       8.744E-03,  TOTAL TIME COMPLETED        8.744E-03


  INCREMENT   110 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.273E-03   TIME AVG. FORCE       6.050E-03
 LARGEST RESIDUAL FORCE             2.937E-14   AT NODE        490   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.686E-05   AT NODE         91   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.082E-05   AT NODE        222   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.080E-02   TIME AVG. MOMENT      5.274E-02
 LARGEST RESIDUAL MOMENT            2.778E-14   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.116E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.471E-07   AT NODE         34   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   8.590E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.027E-02 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.884    
 STEP TIME COMPLETED       8.844E-03,  TOTAL TIME COMPLETED        8.844E-03


  INCREMENT   111 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.244E-03   TIME AVG. FORCE       6.006E-03
 LARGEST RESIDUAL FORCE             4.011E-14   AT NODE         99   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.862E-05   AT NODE        374   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.175E-05   AT NODE        213   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.135E-02   TIME AVG. MOMENT      5.264E-02
 LARGEST RESIDUAL MOMENT            3.083E-14   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.285E-06   AT NODE        221   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.588E-07   AT NODE        202   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -7.927E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   9.463E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.894    
 STEP TIME COMPLETED       8.944E-03,  TOTAL TIME COMPLETED        8.944E-03


  INCREMENT   112 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.315E-03   TIME AVG. FORCE       5.964E-03
 LARGEST RESIDUAL FORCE            -2.907E-14   AT NODE         84   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.227E-05   AT NODE         68   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.291E-05   AT NODE         27   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.254E-02   TIME AVG. MOMENT      5.255E-02
 LARGEST RESIDUAL MOMENT            2.309E-14   AT NODE        465   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.365E-06   AT NODE         29   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.045E-07   AT NODE        194   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   7.369E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -8.649E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.904    
 STEP TIME COMPLETED       9.044E-03,  TOTAL TIME COMPLETED        9.044E-03


  INCREMENT   113 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.453E-03   TIME AVG. FORCE       5.924E-03
 LARGEST RESIDUAL FORCE            -3.237E-14   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.232E-05   AT NODE         67   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.279E-05   AT NODE        198   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.443E-02   TIME AVG. MOMENT      5.247E-02
 LARGEST RESIDUAL MOMENT            3.543E-14   AT NODE        487   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.283E-06   AT NODE        206   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.863E-07   AT NODE         28   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -6.874E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   7.839E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.914    
 STEP TIME COMPLETED       9.144E-03,  TOTAL TIME COMPLETED        9.144E-03


  INCREMENT   114 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.426E-03   TIME AVG. FORCE       5.885E-03
 LARGEST RESIDUAL FORCE             3.246E-14   AT NODE        473   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.464E-05   AT NODE        186   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.195E-05   AT NODE        190   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.536E-02   TIME AVG. MOMENT      5.241E-02
 LARGEST RESIDUAL MOMENT           -2.314E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.258E-06   AT NODE        199   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.539E-07   AT NODE        201   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   6.384E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -7.093E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.924    
 STEP TIME COMPLETED       9.244E-03,  TOTAL TIME COMPLETED        9.244E-03


  INCREMENT   115 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.192E-03   TIME AVG. FORCE       5.844E-03
 LARGEST RESIDUAL FORCE             4.396E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.547E-05   AT NODE        180   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.131E-05   AT NODE        182   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.522E-02   TIME AVG. MOMENT      5.235E-02
 LARGEST RESIDUAL MOMENT            2.320E-14   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.231E-06   AT NODE        192   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.526E-07   AT NODE         24   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -5.859E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   6.444E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.934    
 STEP TIME COMPLETED       9.344E-03,  TOTAL TIME COMPLETED        9.344E-03


  INCREMENT   116 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.383E-03   TIME AVG. FORCE       5.806E-03
 LARGEST RESIDUAL FORCE            -4.565E-14   AT NODE        483   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.305E-05   AT NODE        173   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.439E-05   AT NODE        125   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.924E-02   TIME AVG. MOMENT      5.232E-02
 LARGEST RESIDUAL MOMENT           -3.065E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.288E-06   AT NODE        159   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.519E-07   AT NODE         10   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   5.324E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -5.870E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.944    
 STEP TIME COMPLETED       9.444E-03,  TOTAL TIME COMPLETED        9.444E-03


  INCREMENT   117 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.482E-03   TIME AVG. FORCE       5.769E-03
 LARGEST RESIDUAL FORCE             3.127E-14   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.646E-05   AT NODE         17   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.376E-05   AT NODE        122   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.759E-02   TIME AVG. MOMENT      5.228E-02
 LARGEST RESIDUAL MOMENT            2.688E-14   AT NODE        486   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -9.732E-07   AT NODE        151   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.858E-07   AT NODE          9   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.847E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   5.315E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.954    
 STEP TIME COMPLETED       9.544E-03,  TOTAL TIME COMPLETED        9.544E-03


  INCREMENT   118 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.617E-04   TIME AVG. FORCE       5.723E-03
 LARGEST RESIDUAL FORCE            -3.851E-14   AT NODE        494   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.052E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        6.809E-06   AT NODE         15   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.823E-02   TIME AVG. MOMENT      5.216E-02
 LARGEST RESIDUAL MOMENT            3.202E-14   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.538E-06   AT NODE        139   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.426E-07   AT NODE         12   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   4.462E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -4.757E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.964    
 STEP TIME COMPLETED       9.644E-03,  TOTAL TIME COMPLETED        9.644E-03


  INCREMENT   119 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.138E-03   TIME AVG. FORCE       5.684E-03
 LARGEST RESIDUAL FORCE            -4.570E-14   AT NODE        483   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.656E-05   AT NODE        150   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.009E-05   AT NODE        124   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.630E-02   TIME AVG. MOMENT      5.203E-02
 LARGEST RESIDUAL MOMENT            2.963E-14   AT NODE        489   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.095E-06   AT NODE        137   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.455E-07   AT NODE        136   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.089E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   4.250E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.974    
 STEP TIME COMPLETED       9.744E-03,  TOTAL TIME COMPLETED        9.744E-03


  INCREMENT   120 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.325E-03   TIME AVG. FORCE       5.648E-03
 LARGEST RESIDUAL FORCE             4.748E-14   AT NODE         96   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.241E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.987E-05   AT NODE        122   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.856E-02   TIME AVG. MOMENT      5.192E-02
 LARGEST RESIDUAL MOMENT            3.352E-14   AT NODE        482   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.131E-06   AT NODE        137   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.222E-07   AT NODE          9   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   3.687E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -3.823E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.984    
 STEP TIME COMPLETED       9.844E-03,  TOTAL TIME COMPLETED        9.844E-03


  INCREMENT   121 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.153E-03   TIME AVG. FORCE       5.611E-03
 LARGEST RESIDUAL FORCE             5.143E-14   AT NODE        482   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.053E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        7.451E-06   AT NODE        209   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.836E-02   TIME AVG. MOMENT      5.180E-02
 LARGEST RESIDUAL MOMENT            3.882E-14   AT NODE        483   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      9.085E-07   AT NODE        222   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.881E-07   AT NODE        198   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -3.309E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   3.444E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.994    
 STEP TIME COMPLETED       9.944E-03,  TOTAL TIME COMPLETED        9.944E-03


  INCREMENT   122 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.553E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.912E-04   TIME AVG. FORCE       5.573E-03
 LARGEST RESIDUAL FORCE             3.094E-14   AT NODE         91   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -2.846E-05   AT NODE        125   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.335E-06   AT NODE        122   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.375E-02   TIME AVG. MOMENT      5.166E-02
 LARGEST RESIDUAL MOMENT           -2.721E-14   AT NODE        481   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      8.519E-07   AT NODE        136   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.961E-07   AT NODE        133   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.005E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.039E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.553E-05,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED       1.000E-02,  TOTAL TIME COMPLETED        1.000E-02


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF        122  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                     122  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     124  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                      26  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       2  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       0  WARNING MESSAGES DURING USER INPUT PROCESSING
                       0  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       0  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =   3.9000    
       SYSTEM TIME (SEC)    =  0.30000    
       TOTAL CPU TIME (SEC) =   4.2000    
       WALLCLOCK TIME (SEC) =          6
