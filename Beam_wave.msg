1

   Abaqus 2020                                  Date 31-Jan-2024   Time 15:01:19
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

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-06

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.882E-04   TIME AVG. FORCE       3.882E-04
 LARGEST RESIDUAL FORCE             2.168E-19   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.540E-08   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.540E-08   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.480E-06   TIME AVG. MOMENT      7.480E-06
 LARGEST RESIDUAL MOMENT            2.168E-19   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.557E-10   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.557E-10   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   4.240E-03 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -2.658E-04 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      3.721E-03   TIME AVG. FORCE       2.055E-03
 LARGEST RESIDUAL FORCE            -1.735E-18   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.049E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        5.883E-08   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.649E-04   TIME AVG. MOMENT      4.362E-04
 LARGEST RESIDUAL MOMENT            4.337E-19   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.565E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.099E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.403E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.063E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      3.594E-03   TIME AVG. FORCE       2.568E-03
 LARGEST RESIDUAL FORCE             5.204E-18   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         2.506E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.333E-08   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.436E-03   TIME AVG. MOMENT      1.103E-03
 LARGEST RESIDUAL MOMENT            6.858E-18   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      6.676E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.612E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.772E-03 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -6.533E-04 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      9.500E-03   TIME AVG. FORCE       4.301E-03
 LARGEST RESIDUAL FORCE             4.163E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         9.430E-08   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.918E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.785E-03   TIME AVG. MOMENT      1.523E-03
 LARGEST RESIDUAL MOMENT           -1.041E-17   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.754E-08   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.177E-09   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -7.599E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   8.059E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      1.511E-02   TIME AVG. FORCE       6.463E-03
 LARGEST RESIDUAL FORCE             3.469E-17   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -1.622E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -5.909E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.818E-03   TIME AVG. MOMENT      2.782E-03
 LARGEST RESIDUAL MOMENT            2.082E-17   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.224E-08   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.396E-08   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.800E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.079E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.462E-02   TIME AVG. FORCE       9.490E-03
 LARGEST RESIDUAL FORCE            -5.551E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -8.697E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.478E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.133E-02   TIME AVG. MOMENT      4.206E-03
 LARGEST RESIDUAL MOMENT           -1.110E-16   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.235E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.174E-08   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.343     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -7.146E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      1.913E-02   TIME AVG. FORCE       1.087E-02
 LARGEST RESIDUAL FORCE             4.857E-17   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.562E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.701E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.917E-02   TIME AVG. MOMENT      6.343E-03
 LARGEST RESIDUAL MOMENT           -1.211E-16   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.208E-07   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.823E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.303     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   9.406E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      8.530E-03   TIME AVG. FORCE       1.058E-02
 LARGEST RESIDUAL FORCE             6.245E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.490E-06   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -8.415E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.064E-02   TIME AVG. MOMENT      8.130E-03
 LARGEST RESIDUAL MOMENT            1.336E-16   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.771E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.826E-07   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   5.750E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.116     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.638E-02   TIME AVG. FORCE       1.233E-02
 LARGEST RESIDUAL FORCE            -1.110E-16   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -1.060E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -7.078E-06   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.591E-02   TIME AVG. MOMENT      1.122E-02
 LARGEST RESIDUAL MOMENT           -6.661E-16   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.789E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.080E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.169     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.189     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      3.269E-02   TIME AVG. FORCE       1.437E-02
 LARGEST RESIDUAL FORCE            -1.249E-16   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.877E-06   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.704E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.076E-02   TIME AVG. MOMENT      1.517E-02
 LARGEST RESIDUAL MOMENT            6.384E-16   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -8.605E-07   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.194E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.776     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.479     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      4.068E-02   TIME AVG. FORCE       1.676E-02
 LARGEST RESIDUAL FORCE             6.661E-16   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.168E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.834E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.178E-02   TIME AVG. MOMENT      2.123E-02
 LARGEST RESIDUAL MOMENT            1.998E-15   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.156E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.321E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.02     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.904     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      3.069E-02   TIME AVG. FORCE       1.792E-02
 LARGEST RESIDUAL FORCE             1.416E-15   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.739E-05   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.730E-05   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.174       TIME AVG. MOMENT      3.399E-02
 LARGEST RESIDUAL MOMENT           -3.331E-15   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      6.026E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -7.752E-06   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.28     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.44     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      3.801E-02   TIME AVG. FORCE       1.947E-02
 LARGEST RESIDUAL FORCE            -7.459E-16   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.097E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.191E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.163       TIME AVG. MOMENT      4.391E-02
 LARGEST RESIDUAL MOMENT           -5.995E-15   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.905E-06   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.130E-05   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.52     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.82     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      3.193E-02   TIME AVG. FORCE       2.036E-02
 LARGEST RESIDUAL FORCE            -1.192E-15   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.974E-05   AT NODE        100   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -6.355E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.122       TIME AVG. MOMENT      4.950E-02
 LARGEST RESIDUAL MOMENT            3.941E-15   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.369E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.460E-05   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.35     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    2.80     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.655E-02   TIME AVG. FORCE       2.077E-02
 LARGEST RESIDUAL FORCE             2.526E-15   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.426E-05   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        8.967E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.122       TIME AVG. MOMENT      5.430E-02
 LARGEST RESIDUAL MOMENT            5.052E-15   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.789E-06   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.451E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.76     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -3.56     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.253E-02   TIME AVG. FORCE       2.088E-02
 LARGEST RESIDUAL FORCE            -3.131E-15   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.579E-05   AT NODE         99   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -7.712E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.129       TIME AVG. MOMENT      5.895E-02
 LARGEST RESIDUAL MOMENT           -7.716E-15   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.908E-06   AT NODE        491   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.105E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.10     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    5.49     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.236E-02   TIME AVG. FORCE       2.097E-02
 LARGEST RESIDUAL FORCE            -2.151E-15   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.355E-05   AT NODE        491   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.036E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.119       TIME AVG. MOMENT      6.247E-02
 LARGEST RESIDUAL MOMENT            1.116E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.839E-06   AT NODE        487   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.478E-05   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.71     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -7.74     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      1.981E-02   TIME AVG. FORCE       2.090E-02
 LARGEST RESIDUAL FORCE            -2.887E-15   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.208E-05   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.240E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.120       TIME AVG. MOMENT      6.566E-02
 LARGEST RESIDUAL MOMENT            7.994E-15   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.801E-06   AT NODE        483   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.609E-05   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    4.64     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    10.4     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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
       ELAPSED WALLCLOCK TIME (SEC) =          1

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.678E-02   TIME AVG. FORCE       2.069E-02
 LARGEST RESIDUAL FORCE             4.539E-15   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.594E-05   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.280E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.113       TIME AVG. MOMENT      6.815E-02
 LARGEST RESIDUAL MOMENT           -1.088E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.923E-06   AT NODE        479   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.501E-05   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -8.01     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -13.4     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      1.491E-02   TIME AVG. FORCE       2.040E-02
 LARGEST RESIDUAL FORCE            -4.857E-15   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.247E-05   AT NODE        469   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.340E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.107       TIME AVG. MOMENT      7.008E-02
 LARGEST RESIDUAL MOMENT            1.127E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.032E-06   AT NODE        474   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.280E-05   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    12.5     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    17.0     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      1.312E-02   TIME AVG. FORCE       2.005E-02
 LARGEST RESIDUAL FORCE            -1.197E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.177E-05   AT NODE        463   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.296E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     9.945E-02   TIME AVG. MOMENT      7.148E-02
 LARGEST RESIDUAL MOMENT            1.782E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.039E-06   AT NODE        469   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.224E-05   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -18.6     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    23.6     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      1.182E-02   TIME AVG. FORCE       1.968E-02
 LARGEST RESIDUAL FORCE            -3.737E-15   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.349E-05   AT NODE        457   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.142E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     9.305E-02   TIME AVG. MOMENT      7.246E-02
 LARGEST RESIDUAL MOMENT           -1.368E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.659E-06   AT NODE        463   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.047E-05   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    17.6     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -21.0     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      1.035E-02   TIME AVG. FORCE       1.927E-02
 LARGEST RESIDUAL FORCE             4.219E-15   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.238E-05   AT NODE        451   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.142E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.714E-02   TIME AVG. MOMENT      7.310E-02
 LARGEST RESIDUAL MOMENT            1.737E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.176E-06   AT NODE        444   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.031E-05   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -16.5     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    18.4     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      9.505E-03   TIME AVG. FORCE       1.886E-02
 LARGEST RESIDUAL FORCE             4.968E-15   AT NODE         97   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.286E-05   AT NODE        445   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.090E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.065E-02   TIME AVG. MOMENT      7.341E-02
 LARGEST RESIDUAL MOMENT           -1.146E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.461E-06   AT NODE        438   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.891E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    15.3     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -15.9     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      8.526E-03   TIME AVG. FORCE       1.845E-02
 LARGEST RESIDUAL FORCE             5.954E-15   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.603E-05   AT NODE        439   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.030E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.586E-02   TIME AVG. MOMENT      7.351E-02
 LARGEST RESIDUAL MOMENT            1.264E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.395E-06   AT NODE        432   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.813E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -14.1     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    13.5     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      7.819E-03   TIME AVG. FORCE       1.804E-02
 LARGEST RESIDUAL FORCE             6.474E-15   AT NODE        486   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.350E-05   AT NODE        433   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.464E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.210E-02   TIME AVG. MOMENT      7.346E-02
 LARGEST RESIDUAL MOMENT           -1.267E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.477E-06   AT NODE         82   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.638E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    12.8     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -13.0     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  5.531E-02
 STEP TIME COMPLETED       5.531E-04,  TOTAL TIME COMPLETED        5.531E-04


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.551E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.280E-03   TIME AVG. FORCE       1.764E-02
 LARGEST RESIDUAL FORCE             1.099E-14   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.017E-05   AT NODE        427   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        8.660E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.728E-02   TIME AVG. MOMENT      7.323E-02
 LARGEST RESIDUAL MOMENT           -1.948E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.336E-06   AT NODE        420   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.386E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -11.6     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    13.1     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  6.086E-02
 STEP TIME COMPLETED       6.086E-04,  TOTAL TIME COMPLETED        6.086E-04


  INCREMENT    28 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.551E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.701E-03   TIME AVG. FORCE       1.725E-02
 LARGEST RESIDUAL FORCE            -8.920E-15   AT NODE        100   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.767E-05   AT NODE        406   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -7.814E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.444E-02   TIME AVG. MOMENT      7.291E-02
 LARGEST RESIDUAL MOMENT           -1.101E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.259E-06   AT NODE        414   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.091E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    10.4     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -12.9     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  6.641E-02
 STEP TIME COMPLETED       6.641E-04,  TOTAL TIME COMPLETED        6.641E-04


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.551E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.208E-03   TIME AVG. FORCE       1.687E-02
 LARGEST RESIDUAL FORCE             1.009E-14   AT NODE        100   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.624E-05   AT NODE        400   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        6.860E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.386E-02   TIME AVG. MOMENT      7.260E-02
 LARGEST RESIDUAL MOMENT            9.936E-15   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.822E-06   AT NODE        408   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.791E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -9.27     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    12.6     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    6.939E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  7.196E-02
 STEP TIME COMPLETED       7.196E-04,  TOTAL TIME COMPLETED        7.196E-04


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.939E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.641E-03   TIME AVG. FORCE       1.650E-02
 LARGEST RESIDUAL FORCE            -8.351E-15   AT NODE        497   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.840E-05   AT NODE        394   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -8.428E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.155E-02   TIME AVG. MOMENT      7.223E-02
 LARGEST RESIDUAL MOMENT           -1.141E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.106E-06   AT NODE         76   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.117E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    12.5     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -18.1     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.939E-05,  FRACTION OF STEP COMPLETED  7.890E-02
 STEP TIME COMPLETED       7.890E-04,  TOTAL TIME COMPLETED        7.890E-04


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.939E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.230E-03   TIME AVG. FORCE       1.613E-02
 LARGEST RESIDUAL FORCE             8.538E-15   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.553E-05   AT NODE        388   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        6.649E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.002E-02   TIME AVG. MOMENT      7.184E-02
 LARGEST RESIDUAL MOMENT            1.054E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.889E-06   AT NODE        379   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.631E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -10.8     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    16.8     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.939E-05,  FRACTION OF STEP COMPLETED  8.584E-02
 STEP TIME COMPLETED       8.584E-04,  TOTAL TIME COMPLETED        8.584E-04


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.939E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.843E-03   TIME AVG. FORCE       1.578E-02
 LARGEST RESIDUAL FORCE            -1.150E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.071E-05   AT NODE        381   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -5.664E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.999E-02   TIME AVG. MOMENT      7.147E-02
 LARGEST RESIDUAL MOMENT           -1.492E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.865E-06   AT NODE        372   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.435E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    9.30     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -15.5     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.939E-05,  FRACTION OF STEP COMPLETED  9.278E-02
 STEP TIME COMPLETED       9.278E-04,  TOTAL TIME COMPLETED        9.278E-04


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.939E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.470E-03   TIME AVG. FORCE       1.544E-02
 LARGEST RESIDUAL FORCE            -9.118E-15   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.101E-05   AT NODE        374   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.712E-05   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.539E-02   TIME AVG. MOMENT      7.098E-02
 LARGEST RESIDUAL MOMENT           -7.855E-15   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.890E-06   AT NODE         67   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.333E-05   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    7.97     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    14.2     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    8.674E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.939E-05,  FRACTION OF STEP COMPLETED  0.100    
 STEP TIME COMPLETED       9.972E-04,  TOTAL TIME COMPLETED        9.972E-04


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.674E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.157E-03   TIME AVG. FORCE       1.511E-02
 LARGEST RESIDUAL FORCE             9.022E-15   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.352E-05   AT NODE        367   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        6.193E-05   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.436E-02   TIME AVG. MOMENT      7.049E-02
 LARGEST RESIDUAL MOMENT           -7.973E-15   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.267E-06   AT NODE        358   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.787E-05   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -11.1     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -19.7     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.674E-05,  FRACTION OF STEP COMPLETED  0.108    
 STEP TIME COMPLETED       1.084E-03,  TOTAL TIME COMPLETED        1.084E-03


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.674E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.889E-03   TIME AVG. FORCE       1.479E-02
 LARGEST RESIDUAL FORCE             1.468E-14   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.212E-05   AT NODE         61   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -5.132E-05   AT NODE         66   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.187E-02   TIME AVG. MOMENT      6.996E-02
 LARGEST RESIDUAL MOMENT            8.927E-15   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.159E-06   AT NODE        350   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.495E-05   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    9.89     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    17.5     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.674E-05,  FRACTION OF STEP COMPLETED  0.117    
 STEP TIME COMPLETED       1.171E-03,  TOTAL TIME COMPLETED        1.171E-03


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.674E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.611E-03   TIME AVG. FORCE       1.448E-02
 LARGEST RESIDUAL FORCE            -1.273E-14   AT NODE        485   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.072E-05   AT NODE         59   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.777E-05   AT NODE         64   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.069E-02   TIME AVG. MOMENT      6.943E-02
 LARGEST RESIDUAL MOMENT            1.100E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.794E-06   AT NODE        343   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.403E-05   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -8.79     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -15.5     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.674E-05,  FRACTION OF STEP COMPLETED  0.126    
 STEP TIME COMPLETED       1.257E-03,  TOTAL TIME COMPLETED        1.257E-03


  INCREMENT    37 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.674E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.393E-03   TIME AVG. FORCE       1.418E-02
 LARGEST RESIDUAL FORCE             1.731E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.998E-05   AT NODE         57   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.294E-05   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.874E-02   TIME AVG. MOMENT      6.887E-02
 LARGEST RESIDUAL MOMENT            9.736E-15   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.557E-06   AT NODE        335   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.308E-05   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    7.77     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    13.7     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    1.000E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.674E-05,  FRACTION OF STEP COMPLETED  0.134    
 STEP TIME COMPLETED       1.344E-03,  TOTAL TIME COMPLETED        1.344E-03


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


 AVERAGE FORCE                      3.088E-03   TIME AVG. FORCE       1.388E-02
 LARGEST RESIDUAL FORCE             1.175E-14   AT NODE        484   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.541E-05   AT NODE         55   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        5.325E-05   AT NODE         56   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.702E-02   TIME AVG. MOMENT      6.829E-02
 LARGEST RESIDUAL MOMENT           -8.988E-15   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.895E-06   AT NODE        307   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.508E-05   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -8.97     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -15.8     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.144    
 STEP TIME COMPLETED       1.444E-03,  TOTAL TIME COMPLETED        1.444E-03


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


 AVERAGE FORCE                      2.882E-03   TIME AVG. FORCE       1.360E-02
 LARGEST RESIDUAL FORCE            -1.040E-14   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.532E-05   AT NODE        309   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        5.110E-05   AT NODE        313   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.658E-02   TIME AVG. MOMENT      6.773E-02
 LARGEST RESIDUAL MOMENT           -9.624E-15   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.991E-06   AT NODE        299   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.254E-05   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    7.77     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    13.7     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.154    
 STEP TIME COMPLETED       1.544E-03,  TOTAL TIME COMPLETED        1.544E-03


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


 AVERAGE FORCE                      2.708E-03   TIME AVG. FORCE       1.333E-02
 LARGEST RESIDUAL FORCE             1.518E-14   AT NODE        479   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.937E-05   AT NODE        301   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        5.304E-05   AT NODE        304   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.557E-02   TIME AVG. MOMENT      6.718E-02
 LARGEST RESIDUAL MOMENT           -1.112E-14   AT NODE        478   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.003E-06   AT NODE        292   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.131E-05   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -6.70     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -11.8     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.164    
 STEP TIME COMPLETED       1.644E-03,  TOTAL TIME COMPLETED        1.644E-03


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


 AVERAGE FORCE                      2.493E-03   TIME AVG. FORCE       1.307E-02
 LARGEST RESIDUAL FORCE             1.041E-14   AT NODE        483   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.561E-05   AT NODE         49   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        5.118E-05   AT NODE        295   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.481E-02   TIME AVG. MOMENT      6.663E-02
 LARGEST RESIDUAL MOMENT           -1.128E-14   AT NODE        482   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.829E-06   AT NODE        283   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.051E-05   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    5.75     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    10.7     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.174    
 STEP TIME COMPLETED       1.744E-03,  TOTAL TIME COMPLETED        1.744E-03


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


 AVERAGE FORCE                      2.386E-03   TIME AVG. FORCE       1.281E-02
 LARGEST RESIDUAL FORCE             1.807E-14   AT NODE        489   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.934E-05   AT NODE        285   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.343E-05   AT NODE        286   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.489E-02   TIME AVG. MOMENT      6.612E-02
 LARGEST RESIDUAL MOMENT            1.219E-14   AT NODE        475   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.757E-06   AT NODE        274   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.884E-06   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -4.90     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -9.92     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.184    
 STEP TIME COMPLETED       1.844E-03,  TOTAL TIME COMPLETED        1.844E-03


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


 AVERAGE FORCE                      2.301E-03   TIME AVG. FORCE       1.257E-02
 LARGEST RESIDUAL FORCE            -1.332E-14   AT NODE        475   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.836E-05   AT NODE        256   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.001E-05   AT NODE        261   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.381E-02   TIME AVG. MOMENT      6.560E-02
 LARGEST RESIDUAL MOMENT           -1.054E-14   AT NODE        476   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.529E-06   AT NODE        266   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.826E-06   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    4.14     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    9.17     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.194    
 STEP TIME COMPLETED       1.944E-03,  TOTAL TIME COMPLETED        1.944E-03


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


 AVERAGE FORCE                      2.201E-03   TIME AVG. FORCE       1.233E-02
 LARGEST RESIDUAL FORCE            -1.634E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.948E-05   AT NODE        247   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.768E-05   AT NODE        252   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.264E-02   TIME AVG. MOMENT      6.508E-02
 LARGEST RESIDUAL MOMENT           -1.093E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.303E-06   AT NODE         40   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -7.621E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -3.47     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -8.45     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.204    
 STEP TIME COMPLETED       2.044E-03,  TOTAL TIME COMPLETED        2.044E-03


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


 AVERAGE FORCE                      2.150E-03   TIME AVG. FORCE       1.210E-02
 LARGEST RESIDUAL FORCE             2.446E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.021E-05   AT NODE        239   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.797E-05   AT NODE        243   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.208E-02   TIME AVG. MOMENT      6.457E-02
 LARGEST RESIDUAL MOMENT            1.266E-14   AT NODE        477   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.380E-06   AT NODE        229   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.874E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -2.96     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    7.76     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.214    
 STEP TIME COMPLETED       2.144E-03,  TOTAL TIME COMPLETED        2.144E-03


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


 AVERAGE FORCE                      2.141E-03   TIME AVG. FORCE       1.189E-02
 LARGEST RESIDUAL FORCE             1.441E-14   AT NODE        480   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.736E-05   AT NODE        231   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.603E-05   AT NODE        234   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.420E-02   TIME AVG. MOMENT      6.412E-02
 LARGEST RESIDUAL MOMENT            1.393E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.350E-06   AT NODE        220   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.415E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.70     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -7.11     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.224    
 STEP TIME COMPLETED       2.244E-03,  TOTAL TIME COMPLETED        2.244E-03


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


 AVERAGE FORCE                      2.008E-03   TIME AVG. FORCE       1.168E-02
 LARGEST RESIDUAL FORCE             3.089E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.857E-05   AT NODE        223   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.502E-05   AT NODE        225   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.451E-02   TIME AVG. MOMENT      6.371E-02
 LARGEST RESIDUAL MOMENT           -1.520E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.490E-06   AT NODE        212   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.853E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -2.45     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    6.50     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.234    
 STEP TIME COMPLETED       2.344E-03,  TOTAL TIME COMPLETED        2.344E-03


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


 AVERAGE FORCE                      1.984E-03   TIME AVG. FORCE       1.148E-02
 LARGEST RESIDUAL FORCE            -1.329E-14   AT NODE        478   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.344E-05   AT NODE        193   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.258E-05   AT NODE        217   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.509E-02   TIME AVG. MOMENT      6.332E-02
 LARGEST RESIDUAL MOMENT            8.843E-15   AT NODE        474   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.294E-06   AT NODE        204   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.973E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.22     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -5.92     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.244    
 STEP TIME COMPLETED       2.444E-03,  TOTAL TIME COMPLETED        2.444E-03


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


 AVERAGE FORCE                      1.997E-03   TIME AVG. FORCE       1.128E-02
 LARGEST RESIDUAL FORCE            -2.332E-14   AT NODE        492   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.757E-05   AT NODE        185   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.225E-05   AT NODE        189   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.573E-02   TIME AVG. MOMENT      6.296E-02
 LARGEST RESIDUAL MOMENT            1.333E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.330E-06   AT NODE        195   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.206E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -2.00     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    5.39     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.254    
 STEP TIME COMPLETED       2.544E-03,  TOTAL TIME COMPLETED        2.544E-03


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


 AVERAGE FORCE                      1.894E-03   TIME AVG. FORCE       1.109E-02
 LARGEST RESIDUAL FORCE            -2.153E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.460E-05   AT NODE        177   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.194E-05   AT NODE         21   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.366E-02   TIME AVG. MOMENT      6.257E-02
 LARGEST RESIDUAL MOMENT            1.262E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.133E-06   AT NODE        187   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.809E-06   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.79     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -4.88     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.264    
 STEP TIME COMPLETED       2.644E-03,  TOTAL TIME COMPLETED        2.644E-03


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


 AVERAGE FORCE                      1.859E-03   TIME AVG. FORCE       1.091E-02
 LARGEST RESIDUAL FORCE            -1.568E-14   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.332E-05   AT NODE        169   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.852E-05   AT NODE        173   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.402E-02   TIME AVG. MOMENT      6.221E-02
 LARGEST RESIDUAL MOMENT           -1.297E-14   AT NODE        462   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.079E-06   AT NODE        157   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.535E-06   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.60     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    4.42     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.274    
 STEP TIME COMPLETED       2.744E-03,  TOTAL TIME COMPLETED        2.744E-03


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


 AVERAGE FORCE                      1.990E-03   TIME AVG. FORCE       1.074E-02
 LARGEST RESIDUAL FORCE             1.788E-14   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.137E-05   AT NODE        161   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.834E-05   AT NODE        164   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.523E-02   TIME AVG. MOMENT      6.188E-02
 LARGEST RESIDUAL MOMENT           -1.217E-14   AT NODE        470   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.999E-06   AT NODE         13   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.143E-06   AT NODE        491   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.43     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -3.99     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.284    
 STEP TIME COMPLETED       2.844E-03,  TOTAL TIME COMPLETED        2.844E-03


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


 AVERAGE FORCE                      1.558E-03   TIME AVG. FORCE       1.057E-02
 LARGEST RESIDUAL FORCE            -2.185E-14   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.425E-05   AT NODE        151   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.578E-05   AT NODE        155   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.909E-02   TIME AVG. MOMENT      6.145E-02
 LARGEST RESIDUAL MOMENT            1.387E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.084E-06   AT NODE        140   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.024E-06   AT NODE        491   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.37     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    3.60     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.294    
 STEP TIME COMPLETED       2.944E-03,  TOTAL TIME COMPLETED        2.944E-03


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


 AVERAGE FORCE                      1.699E-03   TIME AVG. FORCE       1.040E-02
 LARGEST RESIDUAL FORCE             1.535E-14   AT NODE        487   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.874E-05   AT NODE          6   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.162E-05   AT NODE        123   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.307E-02   TIME AVG. MOMENT      6.111E-02
 LARGEST RESIDUAL MOMENT           -1.393E-14   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.649E-06   AT NODE          9   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.783E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.39     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -3.23     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.304    
 STEP TIME COMPLETED       3.044E-03,  TOTAL TIME COMPLETED        3.044E-03


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


 AVERAGE FORCE                      1.847E-03   TIME AVG. FORCE       1.025E-02
 LARGEST RESIDUAL FORCE            -2.427E-14   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.236E-05   AT NODE        181   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.830E-05   AT NODE        120   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.584E-02   TIME AVG. MOMENT      6.083E-02
 LARGEST RESIDUAL MOMENT            1.381E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.222E-06   AT NODE        190   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.655E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.39     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    2.90     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.314    
 STEP TIME COMPLETED       3.144E-03,  TOTAL TIME COMPLETED        3.144E-03


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


 AVERAGE FORCE                      1.267E-03   TIME AVG. FORCE       1.009E-02
 LARGEST RESIDUAL FORCE             2.500E-14   AT NODE         97   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.728E-05   AT NODE        122   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.865E-05   AT NODE        177   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.839E-02   TIME AVG. MOMENT      6.043E-02
 LARGEST RESIDUAL MOMENT           -1.601E-14   AT NODE         98   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.858E-06   AT NODE        133   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.542E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.38     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -2.59     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.324    
 STEP TIME COMPLETED       3.244E-03,  TOTAL TIME COMPLETED        3.244E-03


  INCREMENT    57 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.567E-03   TIME AVG. FORCE       9.939E-03
 LARGEST RESIDUAL FORCE             2.752E-14   AT NODE         97   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.911E-05   AT NODE        123   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.799E-05   AT NODE        120   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.867E-02   TIME AVG. MOMENT      6.005E-02
 LARGEST RESIDUAL MOMENT            1.609E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.113E-06   AT NODE          4   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.353E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.35     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    2.31     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.334    
 STEP TIME COMPLETED       3.344E-03,  TOTAL TIME COMPLETED        3.344E-03


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


 AVERAGE FORCE                      1.497E-03   TIME AVG. FORCE       9.793E-03
 LARGEST RESIDUAL FORCE            -1.921E-14   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.318E-05   AT NODE        157   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.346E-05   AT NODE        155   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.997E-02   TIME AVG. MOMENT      5.970E-02
 LARGEST RESIDUAL MOMENT           -1.322E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.704E-06   AT NODE         12   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.157E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.31     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -2.06     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.344    
 STEP TIME COMPLETED       3.444E-03,  TOTAL TIME COMPLETED        3.444E-03


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


 AVERAGE FORCE                      1.478E-03   TIME AVG. FORCE       9.652E-03
 LARGEST RESIDUAL FORCE            -1.886E-14   AT NODE         97   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.056E-05   AT NODE        164   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.342E-05   AT NODE        136   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.393E-02   TIME AVG. MOMENT      5.944E-02
 LARGEST RESIDUAL MOMENT           -2.028E-14   AT NODE        486   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.833E-06   AT NODE        179   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.011E-06   AT NODE        127   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.27     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.83     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.354    
 STEP TIME COMPLETED       3.544E-03,  TOTAL TIME COMPLETED        3.544E-03


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


 AVERAGE FORCE                      1.417E-03   TIME AVG. FORCE       9.515E-03
 LARGEST RESIDUAL FORCE            -3.742E-14   AT NODE        479   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.591E-05   AT NODE         25   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.615E-05   AT NODE        169   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.354E-02   TIME AVG. MOMENT      5.917E-02
 LARGEST RESIDUAL MOMENT            1.368E-14   AT NODE        478   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.548E-06   AT NODE        160   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.768E-06   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.21     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.69     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.364    
 STEP TIME COMPLETED       3.644E-03,  TOTAL TIME COMPLETED        3.644E-03


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


 AVERAGE FORCE                      1.415E-03   TIME AVG. FORCE       9.382E-03
 LARGEST RESIDUAL FORCE             3.029E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.239E-05   AT NODE          9   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.736E-05   AT NODE          9   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.063E-02   TIME AVG. MOMENT      5.887E-02
 LARGEST RESIDUAL MOMENT           -2.094E-14   AT NODE        484   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.510E-06   AT NODE        191   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.681E-06   AT NODE        142   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.16     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.55     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.374    
 STEP TIME COMPLETED       3.744E-03,  TOTAL TIME COMPLETED        3.744E-03


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


 AVERAGE FORCE                      1.223E-03   TIME AVG. FORCE       9.251E-03
 LARGEST RESIDUAL FORCE             1.441E-14   AT NODE        487   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.440E-05   AT NODE        167   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.785E-05   AT NODE        183   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.444E-02   TIME AVG. MOMENT      5.847E-02
 LARGEST RESIDUAL MOMENT           -1.764E-14   AT NODE        481   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.149E-06   AT NODE        155   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.698E-06   AT NODE        174   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.10     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.42     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.384    
 STEP TIME COMPLETED       3.844E-03,  TOTAL TIME COMPLETED        3.844E-03


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


 AVERAGE FORCE                      1.279E-03   TIME AVG. FORCE       9.124E-03
 LARGEST RESIDUAL FORCE             1.863E-14   AT NODE        485   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.050E-05   AT NODE        175   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.202E-05   AT NODE        194   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.558E-02   TIME AVG. MOMENT      5.811E-02
 LARGEST RESIDUAL MOMENT           -1.771E-14   AT NODE        484   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.957E-06   AT NODE        186   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.367E-06   AT NODE        204   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.04     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.30     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.394    
 STEP TIME COMPLETED       3.944E-03,  TOTAL TIME COMPLETED        3.944E-03


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


 AVERAGE FORCE                      1.375E-03   TIME AVG. FORCE       9.003E-03
 LARGEST RESIDUAL FORCE            -3.179E-14   AT NODE        477   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.255E-05   AT NODE         21   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.426E-05   AT NODE        179   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.948E-02   TIME AVG. MOMENT      5.782E-02
 LARGEST RESIDUAL MOMENT           -2.354E-14   AT NODE         95   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.052E-06   AT NODE        194   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.350E-06   AT NODE        170   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.974     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.18     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.404    
 STEP TIME COMPLETED       4.044E-03,  TOTAL TIME COMPLETED        4.044E-03


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


 AVERAGE FORCE                      1.394E-03   TIME AVG. FORCE       8.886E-03
 LARGEST RESIDUAL FORCE            -2.514E-14   AT NODE         97   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.743E-05   AT NODE        187   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.626E-05   AT NODE        207   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.184E-02   TIME AVG. MOMENT      5.757E-02
 LARGEST RESIDUAL MOMENT            1.361E-14   AT NODE        485   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.150E-06   AT NODE        201   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.527E-06   AT NODE        199   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.912     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.08     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.414    
 STEP TIME COMPLETED       4.144E-03,  TOTAL TIME COMPLETED        4.144E-03


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


 AVERAGE FORCE                      1.506E-03   TIME AVG. FORCE       8.774E-03
 LARGEST RESIDUAL FORCE            -1.842E-14   AT NODE        475   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.516E-05   AT NODE         31   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.300E-05   AT NODE        219   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.286E-02   TIME AVG. MOMENT      5.735E-02
 LARGEST RESIDUAL MOMENT           -1.516E-14   AT NODE         98   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.968E-06   AT NODE         28   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.361E-06   AT NODE        227   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.851     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.977     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.424    
 STEP TIME COMPLETED       4.244E-03,  TOTAL TIME COMPLETED        4.244E-03


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


 AVERAGE FORCE                      1.443E-03   TIME AVG. FORCE       8.665E-03
 LARGEST RESIDUAL FORCE            -2.683E-14   AT NODE         97   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.430E-05   AT NODE        229   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.996E-05   AT NODE        247   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.549E-02   TIME AVG. MOMENT      5.717E-02
 LARGEST RESIDUAL MOMENT            2.180E-14   AT NODE        479   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.856E-06   AT NODE        241   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.075E-06   AT NODE        197   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.791     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.884     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.434    
 STEP TIME COMPLETED       4.344E-03,  TOTAL TIME COMPLETED        4.344E-03


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


 AVERAGE FORCE                      1.442E-03   TIME AVG. FORCE       8.559E-03
 LARGEST RESIDUAL FORCE            -2.410E-14   AT NODE        485   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.870E-05   AT NODE        236   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.985E-05   AT NODE        233   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.449E-02   TIME AVG. MOMENT      5.699E-02
 LARGEST RESIDUAL MOMENT           -2.179E-14   AT NODE         97   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.871E-06   AT NODE         38   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.153E-06   AT NODE        224   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.733     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.798     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.444    
 STEP TIME COMPLETED       4.444E-03,  TOTAL TIME COMPLETED        4.444E-03


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


 AVERAGE FORCE                      1.504E-03   TIME AVG. FORCE       8.456E-03
 LARGEST RESIDUAL FORCE            -2.895E-14   AT NODE        470   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.271E-05   AT NODE        270   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.976E-05   AT NODE        242   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.266E-02   TIME AVG. MOMENT      5.678E-02
 LARGEST RESIDUAL MOMENT           -2.522E-14   AT NODE        482   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.816E-06   AT NODE         40   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.024E-06   AT NODE        251   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.676     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.718     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.454    
 STEP TIME COMPLETED       4.544E-03,  TOTAL TIME COMPLETED        4.544E-03


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


 AVERAGE FORCE                      1.373E-03   TIME AVG. FORCE       8.355E-03
 LARGEST RESIDUAL FORCE             3.125E-14   AT NODE        483   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.645E-05   AT NODE         45   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.043E-05   AT NODE        271   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.183E-02   TIME AVG. MOMENT      5.657E-02
 LARGEST RESIDUAL MOMENT           -1.671E-14   AT NODE        478   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.816E-06   AT NODE        265   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.179E-06   AT NODE         41   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.623     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.644     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.464    
 STEP TIME COMPLETED       4.644E-03,  TOTAL TIME COMPLETED        4.644E-03


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


 AVERAGE FORCE                      1.266E-03   TIME AVG. FORCE       8.255E-03
 LARGEST RESIDUAL FORCE             1.887E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.097E-05   AT NODE        285   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.937E-05   AT NODE        281   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.909E-02   TIME AVG. MOMENT      5.632E-02
 LARGEST RESIDUAL MOMENT            1.704E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.699E-06   AT NODE        273   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     9.758E-07   AT NODE        272   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.572     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.577     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.474    
 STEP TIME COMPLETED       4.744E-03,  TOTAL TIME COMPLETED        4.744E-03


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


 AVERAGE FORCE                      1.285E-03   TIME AVG. FORCE       8.159E-03
 LARGEST RESIDUAL FORCE            -2.045E-14   AT NODE         95   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.340E-05   AT NODE        293   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.092E-05   AT NODE         48   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.646E-02   TIME AVG. MOMENT      5.604E-02
 LARGEST RESIDUAL MOMENT           -1.650E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.613E-06   AT NODE        304   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.076E-06   AT NODE         50   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.523     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.515     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.484    
 STEP TIME COMPLETED       4.844E-03,  TOTAL TIME COMPLETED        4.844E-03


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


 AVERAGE FORCE                      1.251E-03   TIME AVG. FORCE       8.064E-03
 LARGEST RESIDUAL FORCE             2.702E-14   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.269E-05   AT NODE        300   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.928E-05   AT NODE        298   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.472E-02   TIME AVG. MOMENT      5.575E-02
 LARGEST RESIDUAL MOMENT            2.127E-14   AT NODE        489   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.617E-06   AT NODE        312   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.032E-06   AT NODE        307   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.478     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.458     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.494    
 STEP TIME COMPLETED       4.944E-03,  TOTAL TIME COMPLETED        4.944E-03


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


 AVERAGE FORCE                      1.359E-03   TIME AVG. FORCE       7.973E-03
 LARGEST RESIDUAL FORCE             1.828E-14   AT NODE        485   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.737E-05   AT NODE        307   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.707E-05   AT NODE         52   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.382E-02   TIME AVG. MOMENT      5.546E-02
 LARGEST RESIDUAL MOMENT            1.828E-14   AT NODE         97   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.571E-06   AT NODE        320   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.510E-07   AT NODE        317   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.435     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.412     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.504    
 STEP TIME COMPLETED       5.044E-03,  TOTAL TIME COMPLETED        5.044E-03


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


 AVERAGE FORCE                      1.390E-03   TIME AVG. FORCE       7.886E-03
 LARGEST RESIDUAL FORCE             2.381E-14   AT NODE        470   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.185E-05   AT NODE        315   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.654E-05   AT NODE        335   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.490E-02   TIME AVG. MOMENT      5.518E-02
 LARGEST RESIDUAL MOMENT            1.692E-14   AT NODE        474   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.601E-06   AT NODE        329   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.663E-07   AT NODE         57   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.395     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.385     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.514    
 STEP TIME COMPLETED       5.144E-03,  TOTAL TIME COMPLETED        5.144E-03


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


 AVERAGE FORCE                      1.456E-03   TIME AVG. FORCE       7.801E-03
 LARGEST RESIDUAL FORCE             2.988E-14   AT NODE        473   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.702E-05   AT NODE        206   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.831E-05   AT NODE        344   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.812E-02   TIME AVG. MOMENT      5.496E-02
 LARGEST RESIDUAL MOMENT           -2.426E-14   AT NODE        479   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.600E-06   AT NODE        336   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.595E-07   AT NODE        334   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.358     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.359     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.524    
 STEP TIME COMPLETED       5.244E-03,  TOTAL TIME COMPLETED        5.244E-03


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


 AVERAGE FORCE                      1.440E-03   TIME AVG. FORCE       7.718E-03
 LARGEST RESIDUAL FORCE             2.891E-14   AT NODE        471   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.575E-05   AT NODE        357   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.731E-05   AT NODE        352   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.016E-02   TIME AVG. MOMENT      5.477E-02
 LARGEST RESIDUAL MOMENT            2.059E-14   AT NODE        488   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.481E-06   AT NODE        344   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.755E-07   AT NODE        362   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.323     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.334     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.534    
 STEP TIME COMPLETED       5.344E-03,  TOTAL TIME COMPLETED        5.344E-03


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


 AVERAGE FORCE                      1.502E-03   TIME AVG. FORCE       7.639E-03
 LARGEST RESIDUAL FORCE            -2.688E-14   AT NODE        481   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.782E-05   AT NODE        364   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.753E-05   AT NODE        361   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.372E-02   TIME AVG. MOMENT      5.462E-02
 LARGEST RESIDUAL MOMENT           -1.408E-14   AT NODE        465   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.463E-06   AT NODE        352   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.216E-07   AT NODE        370   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.291     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.309     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.544    
 STEP TIME COMPLETED       5.444E-03,  TOTAL TIME COMPLETED        5.444E-03


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


 AVERAGE FORCE                      1.531E-03   TIME AVG. FORCE       7.561E-03
 LARGEST RESIDUAL FORCE             2.464E-14   AT NODE         96   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.796E-05   AT NODE        372   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.660E-05   AT NODE        391   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.543E-02   TIME AVG. MOMENT      5.451E-02
 LARGEST RESIDUAL MOMENT            2.007E-14   AT NODE        471   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.402E-06   AT NODE        384   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.044E-07   AT NODE        378   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.261     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.285     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.554    
 STEP TIME COMPLETED       5.544E-03,  TOTAL TIME COMPLETED        5.544E-03


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


 AVERAGE FORCE                      1.470E-03   TIME AVG. FORCE       7.485E-03
 LARGEST RESIDUAL FORCE            -3.186E-14   AT NODE        485   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.903E-05   AT NODE        428   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.582E-05   AT NODE        398   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.393E-02   TIME AVG. MOMENT      5.438E-02
 LARGEST RESIDUAL MOMENT           -2.067E-14   AT NODE        466   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.414E-06   AT NODE        416   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.898E-07   AT NODE        388   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.234     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.263     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.564    
 STEP TIME COMPLETED       5.644E-03,  TOTAL TIME COMPLETED        5.644E-03


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


 AVERAGE FORCE                      1.492E-03   TIME AVG. FORCE       7.411E-03
 LARGEST RESIDUAL FORCE             2.523E-14   AT NODE        474   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.976E-05   AT NODE        387   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.766E-05   AT NODE        407   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.439E-02   TIME AVG. MOMENT      5.425E-02
 LARGEST RESIDUAL MOMENT            2.024E-14   AT NODE         97   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.487E-06   AT NODE        399   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.666E-07   AT NODE        418   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.209     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.241     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.574    
 STEP TIME COMPLETED       5.744E-03,  TOTAL TIME COMPLETED        5.744E-03


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


 AVERAGE FORCE                      1.538E-03   TIME AVG. FORCE       7.340E-03
 LARGEST RESIDUAL FORCE            -3.018E-14   AT NODE        465   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.893E-05   AT NODE        394   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.602E-05   AT NODE         79   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.563E-02   TIME AVG. MOMENT      5.415E-02
 LARGEST RESIDUAL MOMENT           -2.477E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.327E-06   AT NODE        407   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.901E-07   AT NODE        405   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.187     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.221     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.584    
 STEP TIME COMPLETED       5.844E-03,  TOTAL TIME COMPLETED        5.844E-03


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


 AVERAGE FORCE                      1.390E-03   TIME AVG. FORCE       7.268E-03
 LARGEST RESIDUAL FORCE            -2.802E-14   AT NODE        474   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.997E-05   AT NODE        452   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.660E-05   AT NODE        423   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.187E-02   TIME AVG. MOMENT      5.400E-02
 LARGEST RESIDUAL MOMENT           -2.028E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.460E-06   AT NODE        415   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.207E-07   AT NODE         79   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.166     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.201     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.594    
 STEP TIME COMPLETED       5.944E-03,  TOTAL TIME COMPLETED        5.944E-03


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


 AVERAGE FORCE                      1.309E-03   TIME AVG. FORCE       7.197E-03
 LARGEST RESIDUAL FORCE             2.749E-14   AT NODE        471   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.089E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.608E-05   AT NODE        431   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.296E-02   TIME AVG. MOMENT      5.387E-02
 LARGEST RESIDUAL MOMENT           -1.785E-14   AT NODE        453   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.868E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.110E-07   AT NODE        421   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.147     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.183     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.604    
 STEP TIME COMPLETED       6.044E-03,  TOTAL TIME COMPLETED        6.044E-03


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


 AVERAGE FORCE                      1.471E-03   TIME AVG. FORCE       7.130E-03
 LARGEST RESIDUAL FORCE             3.176E-14   AT NODE        482   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.732E-05   AT NODE         92   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.354E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.637E-02   TIME AVG. MOMENT      5.378E-02
 LARGEST RESIDUAL MOMENT            2.376E-14   AT NODE        483   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.262E-06   AT NODE        453   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.332E-06   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.130     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.167     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.614    
 STEP TIME COMPLETED       6.144E-03,  TOTAL TIME COMPLETED        6.144E-03


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


 AVERAGE FORCE                      1.231E-03   TIME AVG. FORCE       7.061E-03
 LARGEST RESIDUAL FORCE            -2.806E-14   AT NODE        476   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.943E-05   AT NODE        478   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.178E-05   AT NODE        446   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.020E-02   TIME AVG. MOMENT      5.362E-02
 LARGEST RESIDUAL MOMENT           -1.991E-14   AT NODE        477   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.513E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.305E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.115     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.151     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.624    
 STEP TIME COMPLETED       6.244E-03,  TOTAL TIME COMPLETED        6.244E-03


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


 AVERAGE FORCE                      3.213E-04   TIME AVG. FORCE       6.984E-03
 LARGEST RESIDUAL FORCE             2.692E-14   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.629E-05   AT NODE        482   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.527E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.691E-02   TIME AVG. MOMENT      5.343E-02
 LARGEST RESIDUAL MOMENT           -1.896E-14   AT NODE         97   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.500E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.260E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.101     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.136     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.634    
 STEP TIME COMPLETED       6.344E-03,  TOTAL TIME COMPLETED        6.344E-03


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


 AVERAGE FORCE                      1.197E-03   TIME AVG. FORCE       6.918E-03
 LARGEST RESIDUAL FORCE             2.104E-14   AT NODE        474   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.083E-05   AT NODE        484   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.749E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.975E-02   TIME AVG. MOMENT      5.327E-02
 LARGEST RESIDUAL MOMENT           -1.774E-14   AT NODE         92   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.243E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.059E-06   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -8.787E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.123     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.644    
 STEP TIME COMPLETED       6.444E-03,  TOTAL TIME COMPLETED        6.444E-03


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


 AVERAGE FORCE                      1.094E-03   TIME AVG. FORCE       6.852E-03
 LARGEST RESIDUAL FORCE            -3.603E-14   AT NODE        489   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.763E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.714E-06   AT NODE         81   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.862E-02   TIME AVG. MOMENT      5.311E-02
 LARGEST RESIDUAL MOMENT            1.879E-14   AT NODE        488   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.448E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.325E-07   AT NODE        412   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   7.726E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.110     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.654    
 STEP TIME COMPLETED       6.544E-03,  TOTAL TIME COMPLETED        6.544E-03


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


 AVERAGE FORCE                      1.013E-03   TIME AVG. FORCE       6.787E-03
 LARGEST RESIDUAL FORCE            -2.566E-14   AT NODE        461   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         9.604E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.181E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.255E-02   TIME AVG. MOMENT      5.288E-02
 LARGEST RESIDUAL MOMENT           -2.526E-14   AT NODE        469   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.776E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.809E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -6.706E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   9.910E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.664    
 STEP TIME COMPLETED       6.644E-03,  TOTAL TIME COMPLETED        6.644E-03


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


 AVERAGE FORCE                      9.442E-04   TIME AVG. FORCE       6.723E-03
 LARGEST RESIDUAL FORCE            -3.394E-14   AT NODE        450   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         8.051E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.235E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.111E-02   TIME AVG. MOMENT      5.264E-02
 LARGEST RESIDUAL MOMENT            2.535E-14   AT NODE         88   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.650E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.702E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   5.762E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -8.899E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.674    
 STEP TIME COMPLETED       6.744E-03,  TOTAL TIME COMPLETED        6.744E-03


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


 AVERAGE FORCE                      1.114E-03   TIME AVG. FORCE       6.662E-03
 LARGEST RESIDUAL FORCE            -3.800E-14   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.632E-05   AT NODE         87   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.832E-05   AT NODE        447   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.873E-02   TIME AVG. MOMENT      5.249E-02
 LARGEST RESIDUAL MOMENT           -2.774E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.713E-06   AT NODE        458   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.161E-07   AT NODE         85   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -5.000E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   7.930E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.684    
 STEP TIME COMPLETED       6.844E-03,  TOTAL TIME COMPLETED        6.844E-03


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


 AVERAGE FORCE                      1.274E-03   TIME AVG. FORCE       6.604E-03
 LARGEST RESIDUAL FORCE             3.264E-14   AT NODE        456   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.512E-05   AT NODE        441   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.686E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.152E-02   TIME AVG. MOMENT      5.237E-02
 LARGEST RESIDUAL MOMENT           -2.123E-14   AT NODE         92   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.325E-06   AT NODE        401   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.229E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   4.315E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -7.049E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.694    
 STEP TIME COMPLETED       6.944E-03,  TOTAL TIME COMPLETED        6.944E-03


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


 AVERAGE FORCE                      1.145E-03   TIME AVG. FORCE       6.546E-03
 LARGEST RESIDUAL FORCE            -4.029E-14   AT NODE        446   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.578E-05   AT NODE         94   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.450E-05   AT NODE        412   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.013E-02   TIME AVG. MOMENT      5.224E-02
 LARGEST RESIDUAL MOMENT            1.857E-14   AT NODE        465   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.048E-06   AT NODE         91   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.352E-07   AT NODE        422   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -3.662E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   6.283E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.704    
 STEP TIME COMPLETED       7.044E-03,  TOTAL TIME COMPLETED        7.044E-03


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


 AVERAGE FORCE                      1.098E-03   TIME AVG. FORCE       6.489E-03
 LARGEST RESIDUAL FORCE            -4.740E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.581E-05   AT NODE        472   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.429E-05   AT NODE         88   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.718E-02   TIME AVG. MOMENT      5.208E-02
 LARGEST RESIDUAL MOMENT           -3.656E-14   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.367E-06   AT NODE         90   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -7.927E-07   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   3.087E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -5.588E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.714    
 STEP TIME COMPLETED       7.144E-03,  TOTAL TIME COMPLETED        7.144E-03


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


 AVERAGE FORCE                      1.040E-03   TIME AVG. FORCE       6.432E-03
 LARGEST RESIDUAL FORCE             2.464E-14   AT NODE        451   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.484E-05   AT NODE        411   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.641E-05   AT NODE        417   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.689E-02   TIME AVG. MOMENT      5.193E-02
 LARGEST RESIDUAL MOMENT           -2.962E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.165E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.422E-07   AT NODE        426   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   2.646E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   4.929E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.724    
 STEP TIME COMPLETED       7.244E-03,  TOTAL TIME COMPLETED        7.244E-03


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


 AVERAGE FORCE                      1.197E-03   TIME AVG. FORCE       6.378E-03
 LARGEST RESIDUAL FORCE            -2.514E-14   AT NODE        480   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.291E-05   AT NODE        405   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.797E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.144E-02   TIME AVG. MOMENT      5.182E-02
 LARGEST RESIDUAL MOMENT           -2.815E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.844E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.066E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -2.283E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -4.337E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.734    
 STEP TIME COMPLETED       7.344E-03,  TOTAL TIME COMPLETED        7.344E-03


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


 AVERAGE FORCE                      1.328E-03   TIME AVG. FORCE       6.327E-03
 LARGEST RESIDUAL FORCE            -3.557E-14   AT NODE         97   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.463E-05   AT NODE        399   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.587E-05   AT NODE         76   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.461E-02   TIME AVG. MOMENT      5.174E-02
 LARGEST RESIDUAL MOMENT            2.309E-14   AT NODE        485   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.597E-06   AT NODE        386   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -7.855E-07   AT NODE        391   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.995E-02 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   3.834E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.744    
 STEP TIME COMPLETED       7.444E-03,  TOTAL TIME COMPLETED        7.444E-03


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


 AVERAGE FORCE                      1.360E-03   TIME AVG. FORCE       6.277E-03
 LARGEST RESIDUAL FORCE            -3.137E-14   AT NODE        482   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.667E-05   AT NODE        393   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.290E-05   AT NODE         68   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.532E-02   TIME AVG. MOMENT      5.168E-02
 LARGEST RESIDUAL MOMENT            2.440E-14   AT NODE         96   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.635E-06   AT NODE        378   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.715E-07   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.798E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -3.379E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.754    
 STEP TIME COMPLETED       7.544E-03,  TOTAL TIME COMPLETED        7.544E-03


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


 AVERAGE FORCE                      1.329E-03   TIME AVG. FORCE       6.227E-03
 LARGEST RESIDUAL FORCE             3.050E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.380E-05   AT NODE        386   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.315E-05   AT NODE        386   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.658E-02   TIME AVG. MOMENT      5.163E-02
 LARGEST RESIDUAL MOMENT           -1.824E-14   AT NODE        463   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.405E-06   AT NODE        371   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.584E-07   AT NODE        396   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.641E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   2.951E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.764    
 STEP TIME COMPLETED       7.644E-03,  TOTAL TIME COMPLETED        7.644E-03


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


 AVERAGE FORCE                      1.291E-03   TIME AVG. FORCE       6.178E-03
 LARGEST RESIDUAL FORCE             3.162E-14   AT NODE        447   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.939E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.426E-05   AT NODE        354   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.491E-02   TIME AVG. MOMENT      5.156E-02
 LARGEST RESIDUAL MOMENT           -2.180E-14   AT NODE        474   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.262E-06   AT NODE        339   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.578E-07   AT NODE        365   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.496E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -2.565E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.774    
 STEP TIME COMPLETED       7.744E-03,  TOTAL TIME COMPLETED        7.744E-03


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


 AVERAGE FORCE                      1.232E-03   TIME AVG. FORCE       6.130E-03
 LARGEST RESIDUAL FORCE            -3.146E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.070E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.116E-05   AT NODE        346   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.870E-02   TIME AVG. MOMENT      5.144E-02
 LARGEST RESIDUAL MOMENT           -3.279E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.597E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.447E-07   AT NODE        334   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.345E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   2.239E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.784    
 STEP TIME COMPLETED       7.844E-03,  TOTAL TIME COMPLETED        7.844E-03


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


 AVERAGE FORCE                      1.189E-03   TIME AVG. FORCE       6.082E-03
 LARGEST RESIDUAL FORCE             4.250E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.588E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.222E-05   AT NODE        339   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.903E-02   TIME AVG. MOMENT      5.131E-02
 LARGEST RESIDUAL MOMENT           -3.274E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.323E-06   AT NODE         50   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.529E-07   AT NODE         63   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.197E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.958E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.794    
 STEP TIME COMPLETED       7.944E-03,  TOTAL TIME COMPLETED        7.944E-03


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


 AVERAGE FORCE                      1.216E-03   TIME AVG. FORCE       6.035E-03
 LARGEST RESIDUAL FORCE            -6.391E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.842E-05   AT NODE         82   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.171E-05   AT NODE        307   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.232E-02   TIME AVG. MOMENT      5.123E-02
 LARGEST RESIDUAL MOMENT            3.936E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.308E-06   AT NODE        290   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.885E-07   AT NODE        318   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.075E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.692E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.804    
 STEP TIME COMPLETED       8.044E-03,  TOTAL TIME COMPLETED        8.044E-03


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


 AVERAGE FORCE                      1.212E-03   TIME AVG. FORCE       5.989E-03
 LARGEST RESIDUAL FORCE            -5.598E-14   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.108E-05   AT NODE        422   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.273E-05   AT NODE        298   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.582E-02   TIME AVG. MOMENT      5.118E-02
 LARGEST RESIDUAL MOMENT           -2.954E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.215E-06   AT NODE        284   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.107E-07   AT NODE        288   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   9.693E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.503E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.814    
 STEP TIME COMPLETED       8.144E-03,  TOTAL TIME COMPLETED        8.144E-03


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


 AVERAGE FORCE                      1.278E-03   TIME AVG. FORCE       5.945E-03
 LARGEST RESIDUAL FORCE            -3.371E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.755E-05   AT NODE        263   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.195E-05   AT NODE        292   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.719E-02   TIME AVG. MOMENT      5.114E-02
 LARGEST RESIDUAL MOMENT           -2.887E-14   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.143E-06   AT NODE        324   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.105E-07   AT NODE        302   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -8.646E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.364E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.824    
 STEP TIME COMPLETED       8.244E-03,  TOTAL TIME COMPLETED        8.244E-03


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


 AVERAGE FORCE                      1.381E-03   TIME AVG. FORCE       5.902E-03
 LARGEST RESIDUAL FORCE            -3.962E-14   AT NODE        494   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.406E-05   AT NODE         51   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.205E-05   AT NODE        283   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.728E-02   TIME AVG. MOMENT      5.110E-02
 LARGEST RESIDUAL MOMENT            4.485E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.148E-06   AT NODE        315   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.076E-07   AT NODE         49   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   7.569E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.223E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.834    
 STEP TIME COMPLETED       8.344E-03,  TOTAL TIME COMPLETED        8.344E-03


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


 AVERAGE FORCE                      1.406E-03   TIME AVG. FORCE       5.860E-03
 LARGEST RESIDUAL FORCE            -4.169E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.015E-05   AT NODE        295   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.111E-05   AT NODE        274   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.451E-02   TIME AVG. MOMENT      5.104E-02
 LARGEST RESIDUAL MOMENT            3.242E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.121E-06   AT NODE        309   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.448E-07   AT NODE        287   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -6.644E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.098E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.844    
 STEP TIME COMPLETED       8.444E-03,  TOTAL TIME COMPLETED        8.444E-03


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


 AVERAGE FORCE                      1.361E-03   TIME AVG. FORCE       5.819E-03
 LARGEST RESIDUAL FORCE            -4.366E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.099E-05   AT NODE         48   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.157E-05   AT NODE        267   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.504E-02   TIME AVG. MOMENT      5.099E-02
 LARGEST RESIDUAL MOMENT            2.609E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.207E-06   AT NODE        276   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.735E-07   AT NODE        278   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   5.896E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   9.945E-03 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.854    
 STEP TIME COMPLETED       8.544E-03,  TOTAL TIME COMPLETED        8.544E-03


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


 AVERAGE FORCE                      1.248E-03   TIME AVG. FORCE       5.777E-03
 LARGEST RESIDUAL FORCE             3.012E-14   AT NODE        473   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.653E-05   AT NODE        284   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.108E-05   AT NODE        259   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.360E-02   TIME AVG. MOMENT      5.092E-02
 LARGEST RESIDUAL MOMENT           -2.787E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.118E-06   AT NODE        269   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.268E-07   AT NODE        248   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -5.217E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   9.057E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.864    
 STEP TIME COMPLETED       8.644E-03,  TOTAL TIME COMPLETED        8.644E-03


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


 AVERAGE FORCE                      1.148E-03   TIME AVG. FORCE       5.736E-03
 LARGEST RESIDUAL FORCE             3.766E-14   AT NODE        494   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.685E-05   AT NODE        463   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.119E-05   AT NODE        226   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.950E-02   TIME AVG. MOMENT      5.082E-02
 LARGEST RESIDUAL MOMENT           -1.784E-14   AT NODE         90   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.086E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.413E-07   AT NODE        238   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   4.518E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -8.408E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.874    
 STEP TIME COMPLETED       8.744E-03,  TOTAL TIME COMPLETED        8.744E-03


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


 AVERAGE FORCE                      1.179E-03   TIME AVG. FORCE       5.695E-03
 LARGEST RESIDUAL FORCE            -4.467E-14   AT NODE        489   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.608E-05   AT NODE        464   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.117E-05   AT NODE        220   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.775E-02   TIME AVG. MOMENT      5.070E-02
 LARGEST RESIDUAL MOMENT           -3.768E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.060E-06   AT NODE        227   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.388E-07   AT NODE        231   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   4.021E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   7.836E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.884    
 STEP TIME COMPLETED       8.844E-03,  TOTAL TIME COMPLETED        8.844E-03


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


 AVERAGE FORCE                      1.096E-03   TIME AVG. FORCE       5.654E-03
 LARGEST RESIDUAL FORCE             6.759E-14   AT NODE        489   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.615E-05   AT NODE        373   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.031E-05   AT NODE        210   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.675E-02   TIME AVG. MOMENT      5.058E-02
 LARGEST RESIDUAL MOMENT            4.636E-14   AT NODE         98   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.182E-06   AT NODE        218   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.962E-07   AT NODE        200   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -3.773E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -7.226E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.894    
 STEP TIME COMPLETED       8.944E-03,  TOTAL TIME COMPLETED        8.944E-03


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


 AVERAGE FORCE                      1.164E-03   TIME AVG. FORCE       5.615E-03
 LARGEST RESIDUAL FORCE            -5.751E-14   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.797E-05   AT NODE        369   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.222E-05   AT NODE        203   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.661E-02   TIME AVG. MOMENT      5.045E-02
 LARGEST RESIDUAL MOMENT           -3.871E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.299E-06   AT NODE        212   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.726E-07   AT NODE         29   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   3.503E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   6.579E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.904    
 STEP TIME COMPLETED       9.044E-03,  TOTAL TIME COMPLETED        9.044E-03


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


 AVERAGE FORCE                      1.370E-03   TIME AVG. FORCE       5.578E-03
 LARGEST RESIDUAL FORCE            -3.121E-14   AT NODE         95   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.813E-05   AT NODE        191   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.097E-05   AT NODE        196   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.213E-02   TIME AVG. MOMENT      5.038E-02
 LARGEST RESIDUAL MOMENT            2.728E-14   AT NODE        436   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.099E-06   AT NODE         27   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.367E-07   AT NODE        185   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -3.234E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -5.987E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.914    
 STEP TIME COMPLETED       9.144E-03,  TOTAL TIME COMPLETED        9.144E-03


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


 AVERAGE FORCE                      1.299E-03   TIME AVG. FORCE       5.541E-03
 LARGEST RESIDUAL FORCE            -6.055E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.205E-05   AT NODE        184   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.006E-05   AT NODE        187   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.371E-02   TIME AVG. MOMENT      5.032E-02
 LARGEST RESIDUAL MOMENT            5.139E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.063E-06   AT NODE        197   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.748E-07   AT NODE        177   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   3.018E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   5.516E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.924    
 STEP TIME COMPLETED       9.244E-03,  TOTAL TIME COMPLETED        9.244E-03


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


 AVERAGE FORCE                      1.155E-03   TIME AVG. FORCE       5.504E-03
 LARGEST RESIDUAL FORCE             4.391E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.529E-05   AT NODE        177   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.022E-05   AT NODE         14   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.559E-02   TIME AVG. MOMENT      5.028E-02
 LARGEST RESIDUAL MOMENT           -3.214E-14   AT NODE        472   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.301E-06   AT NODE        162   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.794E-07   AT NODE        166   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -2.807E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -5.097E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.934    
 STEP TIME COMPLETED       9.344E-03,  TOTAL TIME COMPLETED        9.344E-03


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


 AVERAGE FORCE                      1.471E-03   TIME AVG. FORCE       5.469E-03
 LARGEST RESIDUAL FORCE            -3.744E-14   AT NODE         98   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.131E-05   AT NODE        172   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.461E-05   AT NODE        123   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.092E-02   TIME AVG. MOMENT      5.029E-02
 LARGEST RESIDUAL MOMENT           -2.562E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.193E-06   AT NODE         15   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.837E-07   AT NODE        136   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   2.583E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   4.650E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.944    
 STEP TIME COMPLETED       9.444E-03,  TOTAL TIME COMPLETED        9.444E-03


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


 AVERAGE FORCE                      1.383E-03   TIME AVG. FORCE       5.435E-03
 LARGEST RESIDUAL FORCE            -6.089E-14   AT NODE        485   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.220E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -8.674E-06   AT NODE        143   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.634E-02   TIME AVG. MOMENT      5.026E-02
 LARGEST RESIDUAL MOMENT           -4.485E-14   AT NODE         97   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -7.960E-07   AT NODE        147   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.386E-07   AT NODE        132   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -2.345E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -4.188E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.954    
 STEP TIME COMPLETED       9.544E-03,  TOTAL TIME COMPLETED        9.544E-03


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


 AVERAGE FORCE                      9.010E-04   TIME AVG. FORCE       5.397E-03
 LARGEST RESIDUAL FORCE             4.171E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.201E-05   AT NODE        153   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.720E-06   AT NODE        125   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.693E-02   TIME AVG. MOMENT      5.015E-02
 LARGEST RESIDUAL MOMENT           -2.529E-14   AT NODE        459   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.634E-06   AT NODE        138   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.158E-07   AT NODE        138   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   2.141E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   3.787E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.964    
 STEP TIME COMPLETED       9.644E-03,  TOTAL TIME COMPLETED        9.644E-03


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


 AVERAGE FORCE                      1.193E-03   TIME AVG. FORCE       5.363E-03
 LARGEST RESIDUAL FORCE            -5.151E-14   AT NODE        487   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.311E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.977E-05   AT NODE        122   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.936E-02   TIME AVG. MOMENT      5.006E-02
 LARGEST RESIDUAL MOMENT            3.780E-14   AT NODE        484   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.624E-06   AT NODE         10   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.217E-07   AT NODE        134   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.973E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -3.454E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.974    
 STEP TIME COMPLETED       9.744E-03,  TOTAL TIME COMPLETED        9.744E-03


  INCREMENT   122 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.193E-03   TIME AVG. FORCE       5.328E-03
 LARGEST RESIDUAL FORCE             4.563E-14   AT NODE        485   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.433E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.030E-05   AT NODE        122   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.724E-02   TIME AVG. MOMENT      4.995E-02
 LARGEST RESIDUAL MOMENT            2.842E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      9.437E-07   AT NODE        225   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.955E-07   AT NODE        135   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.917E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   3.147E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.984    
 STEP TIME COMPLETED       9.844E-03,  TOTAL TIME COMPLETED        9.844E-03


  INCREMENT   123 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.060E-03   TIME AVG. FORCE       5.294E-03
 LARGEST RESIDUAL FORCE             4.590E-14   AT NODE        482   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.453E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.367E-06   AT NODE         22   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.454E-02   TIME AVG. MOMENT      4.983E-02
 LARGEST RESIDUAL MOMENT            3.138E-14   AT NODE        490   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -9.018E-07   AT NODE          4   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.535E-07   AT NODE        175   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.844E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -2.853E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.994    
 STEP TIME COMPLETED       9.944E-03,  TOTAL TIME COMPLETED        9.944E-03


  INCREMENT   124 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.585E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.939E-04   TIME AVG. FORCE       5.259E-03
 LARGEST RESIDUAL FORCE             4.815E-14   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -2.478E-05   AT NODE          7   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.244E-06   AT NODE          6   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.260E-02   TIME AVG. MOMENT      4.969E-02
 LARGEST RESIDUAL MOMENT           -2.750E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -6.814E-07   AT NODE        114   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.820E-07   AT NODE        133   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   6.219E-04 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   8.751E-04 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.585E-05,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED       1.000E-02,  TOTAL TIME COMPLETED        1.000E-02


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF        124  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                     124  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                     126  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
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
       USER TIME (SEC)      =   4.5000    
       SYSTEM TIME (SEC)    =  0.10000    
       TOTAL CPU TIME (SEC) =   4.6000    
       WALLCLOCK TIME (SEC) =          8
