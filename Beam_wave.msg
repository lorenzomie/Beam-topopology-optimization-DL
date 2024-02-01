1

   Abaqus 2020                                  Date 31-Jan-2024   Time 22:36:17
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

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.721E-04   TIME AVG. FORCE       3.721E-04
 LARGEST RESIDUAL FORCE             3.469E-18   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.430E-08   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.430E-08   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.387E-06   TIME AVG. MOMENT      7.387E-06
 LARGEST RESIDUAL MOMENT            4.337E-19   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.444E-10   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.444E-10   AT NODE        101   DOF  6
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


 AVERAGE FORCE                      3.721E-03   TIME AVG. FORCE       2.047E-03
 LARGEST RESIDUAL FORCE             1.388E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         9.739E-08   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        5.462E-08   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.109E-04   TIME AVG. MOMENT      4.091E-04
 LARGEST RESIDUAL MOMENT           -1.735E-18   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.451E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.019E-09   AT NODE        101   DOF  6
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


 AVERAGE FORCE                      3.595E-03   TIME AVG. FORCE       2.563E-03
 LARGEST RESIDUAL FORCE             2.082E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         2.327E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.238E-08   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.443E-03   TIME AVG. MOMENT      1.087E-03
 LARGEST RESIDUAL MOMENT            5.204E-18   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      6.185E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.418E-09   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.772E-03 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -6.559E-04 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      8.907E-03   TIME AVG. FORCE       4.149E-03
 LARGEST RESIDUAL FORCE             1.388E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         8.760E-08   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.567E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.663E-03   TIME AVG. MOMENT      1.481E-03
 LARGEST RESIDUAL MOMENT           -1.236E-17   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.623E-08   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.024E-09   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -7.600E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   8.067E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      1.432E-02   TIME AVG. FORCE       6.184E-03
 LARGEST RESIDUAL FORCE             2.776E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -1.507E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -5.485E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.520E-03   TIME AVG. MOMENT      2.689E-03
 LARGEST RESIDUAL MOMENT           -2.082E-17   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.135E-08   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.072E-08   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.800E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.074E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.463E-02   TIME AVG. FORCE       9.258E-03
 LARGEST RESIDUAL FORCE             2.776E-17   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -8.071E-07   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.159E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.064E-02   TIME AVG. MOMENT      4.013E-03
 LARGEST RESIDUAL MOMENT            8.327E-17   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.143E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.888E-08   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.343     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -7.154E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      1.768E-02   TIME AVG. FORCE       1.046E-02
 LARGEST RESIDUAL FORCE             5.551E-17   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.308E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.509E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     1.869E-02   TIME AVG. MOMENT      6.110E-03
 LARGEST RESIDUAL MOMENT           -6.939E-17   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.122E-07   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.611E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.303     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   9.423E-02 AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      8.321E-03   TIME AVG. FORCE       1.019E-02
 LARGEST RESIDUAL FORCE            -3.123E-16   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.384E-06   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -7.814E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     2.113E-02   TIME AVG. MOMENT      7.988E-03
 LARGEST RESIDUAL MOMENT            3.053E-16   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.560E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.625E-07   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   5.749E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.117     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.502E-02   TIME AVG. FORCE       1.184E-02
 LARGEST RESIDUAL FORCE            -3.331E-16   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -9.841E-06   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -6.572E-06   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.492E-02   TIME AVG. MOMENT      1.098E-02
 LARGEST RESIDUAL MOMENT           -8.635E-16   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.496E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.941E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.169     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.190     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      3.293E-02   TIME AVG. FORCE       1.395E-02
 LARGEST RESIDUAL FORCE             3.192E-16   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.437E-06   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.512E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.699E-02   TIME AVG. MOMENT      1.558E-02
 LARGEST RESIDUAL MOMENT            1.242E-15   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -7.853E-07   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.032E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.776     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.476     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      4.005E-02   TIME AVG. FORCE       1.632E-02
 LARGEST RESIDUAL FORCE             1.110E-15   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         2.950E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.564E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.106E-02   TIME AVG. MOMENT      2.153E-02
 LARGEST RESIDUAL MOMENT            3.997E-15   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.931E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.989E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.02     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.896     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      3.219E-02   TIME AVG. FORCE       1.764E-02
 LARGEST RESIDUAL FORCE            -2.637E-16   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.457E-05   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.420E-05   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.192       TIME AVG. MOMENT      3.575E-02
 LARGEST RESIDUAL MOMENT           -3.669E-15   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      5.451E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -7.266E-06   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.27     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.43     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      3.925E-02   TIME AVG. FORCE       1.931E-02
 LARGEST RESIDUAL FORCE             1.027E-15   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.538E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.951E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.191       TIME AVG. MOMENT      4.767E-02
 LARGEST RESIDUAL MOMENT           -1.243E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.954E-06   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.030E-05   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.52     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.85     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      3.647E-02   TIME AVG. FORCE       2.053E-02
 LARGEST RESIDUAL FORCE            -9.267E-16   AT NODE        498   DOF  1
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.544E-05   AT NODE        100   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -5.621E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.145       TIME AVG. MOMENT      5.463E-02
 LARGEST RESIDUAL MOMENT            1.128E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.322E-06   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.437E-05   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.35     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    2.89     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.878E-02   TIME AVG. FORCE       2.108E-02
 LARGEST RESIDUAL FORCE            -1.207E-15   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.752E-05   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        8.727E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.134       TIME AVG. MOMENT      5.995E-02
 LARGEST RESIDUAL MOMENT            7.938E-15   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.731E-06   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.080E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.76     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -3.51     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.499E-02   TIME AVG. FORCE       2.133E-02
 LARGEST RESIDUAL FORCE            -2.630E-15   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.837E-05   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -8.248E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.129       TIME AVG. MOMENT      6.426E-02
 LARGEST RESIDUAL MOMENT           -1.128E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.441E-06   AT NODE        490   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.589E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.11     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    5.35     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.364E-02   TIME AVG. FORCE       2.146E-02
 LARGEST RESIDUAL FORCE            -2.748E-15   AT NODE        100   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.738E-05   AT NODE        490   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.350E-05   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.115       TIME AVG. MOMENT      6.724E-02
 LARGEST RESIDUAL MOMENT            1.581E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.474E-06   AT NODE        486   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.168E-05   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.75     AND OCCURS AT NODE        101 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -7.47     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      2.017E-02   TIME AVG. FORCE       2.139E-02
 LARGEST RESIDUAL FORCE            -5.398E-15   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.753E-05   AT NODE         95   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.170E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     0.110       TIME AVG. MOMENT      6.960E-02
 LARGEST RESIDUAL MOMENT           -2.154E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.708E-06   AT NODE        482   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.467E-05   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    4.64     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    9.93     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      1.711E-02   TIME AVG. FORCE       2.117E-02
 LARGEST RESIDUAL FORCE             4.104E-15   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.912E-05   AT NODE         94   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.270E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     9.926E-02   TIME AVG. MOMENT      7.116E-02
 LARGEST RESIDUAL MOMENT            1.515E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.116E-06   AT NODE        478   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.531E-05   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -8.03     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -12.8     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      1.511E-02   TIME AVG. FORCE       2.086E-02
 LARGEST RESIDUAL FORCE             6.495E-15   AT NODE        497   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.377E-05   AT NODE        469   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.358E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     9.590E-02   TIME AVG. MOMENT      7.240E-02
 LARGEST RESIDUAL MOMENT            1.926E-14   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.033E-06   AT NODE         94   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.478E-05   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    12.6     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -16.8     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
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


 AVERAGE FORCE                      1.329E-02   TIME AVG. FORCE       2.050E-02
 LARGEST RESIDUAL FORCE             4.649E-15   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.751E-05   AT NODE        463   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.351E-04   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     9.339E-02   TIME AVG. MOMENT      7.340E-02
 LARGEST RESIDUAL MOMENT            1.910E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      4.010E-06   AT NODE        469   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.688E-05   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -18.7     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    23.9     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      1.192E-02   TIME AVG. FORCE       2.011E-02
 LARGEST RESIDUAL FORCE            -5.041E-15   AT NODE        100   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.706E-05   AT NODE        457   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.139E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.752E-02   TIME AVG. MOMENT      7.404E-02
 LARGEST RESIDUAL MOMENT           -1.305E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.338E-06   AT NODE        462   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.132E-05   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    17.8     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -21.7     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      1.030E-02   TIME AVG. FORCE       1.969E-02
 LARGEST RESIDUAL FORCE            -4.906E-15   AT NODE        497   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.851E-05   AT NODE        450   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.140E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     8.280E-02   TIME AVG. MOMENT      7.442E-02
 LARGEST RESIDUAL MOMENT            1.568E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.053E-06   AT NODE        445   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.994E-05   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -16.7     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    19.5     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      9.409E-03   TIME AVG. FORCE       1.926E-02
 LARGEST RESIDUAL FORCE            -7.633E-15   AT NODE         99   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.558E-05   AT NODE        445   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.104E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.612E-02   TIME AVG. MOMENT      7.449E-02
 LARGEST RESIDUAL MOMENT            1.697E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.048E-06   AT NODE         85   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.040E-05   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    15.5     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -17.3     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      8.608E-03   TIME AVG. FORCE       1.883E-02
 LARGEST RESIDUAL FORCE             7.508E-15   AT NODE        100   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.921E-05   AT NODE        439   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.052E-04   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     7.269E-02   TIME AVG. MOMENT      7.442E-02
 LARGEST RESIDUAL MOMENT           -2.169E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.127E-06   AT NODE        431   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.027E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -14.3     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    15.1     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      7.973E-03   TIME AVG. FORCE       1.841E-02
 LARGEST RESIDUAL FORCE            -8.223E-15   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.522E-05   AT NODE        433   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.866E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.899E-02   TIME AVG. MOMENT      7.421E-02
 LARGEST RESIDUAL MOMENT            1.664E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.036E-06   AT NODE         82   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.980E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    13.0     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -13.8     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      7.198E-03   TIME AVG. FORCE       1.800E-02
 LARGEST RESIDUAL FORCE             6.533E-15   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.343E-05   AT NODE        426   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        9.087E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.442E-02   TIME AVG. MOMENT      7.385E-02
 LARGEST RESIDUAL MOMENT           -1.357E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -3.131E-06   AT NODE        419   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.866E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -11.7     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    13.9     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

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


 AVERAGE FORCE                      6.747E-03   TIME AVG. FORCE       1.760E-02
 LARGEST RESIDUAL FORCE            -1.625E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.122E-05   AT NODE         77   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -8.153E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.331E-02   TIME AVG. MOMENT      7.347E-02
 LARGEST RESIDUAL MOMENT            1.682E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.770E-06   AT NODE         79   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.709E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    10.5     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -13.7     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    6.649E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.551E-05,  FRACTION OF STEP COMPLETED  6.641E-02
 STEP TIME COMPLETED       6.641E-04,  TOTAL TIME COMPLETED        6.641E-04


  INCREMENT    29 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.649E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.151E-03   TIME AVG. FORCE       1.720E-02
 LARGEST RESIDUAL FORCE            -9.975E-15   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.752E-05   AT NODE        399   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        9.517E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     6.227E-02   TIME AVG. MOMENT      7.309E-02
 LARGEST RESIDUAL MOMENT            2.104E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.819E-06   AT NODE        407   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.029E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -13.2     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    18.4     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.649E-05,  FRACTION OF STEP COMPLETED  7.306E-02
 STEP TIME COMPLETED       7.306E-04,  TOTAL TIME COMPLETED        7.306E-04


  INCREMENT    30 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.649E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.680E-03   TIME AVG. FORCE       1.682E-02
 LARGEST RESIDUAL FORCE             6.210E-15   AT NODE        494   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.945E-05   AT NODE        393   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -7.261E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.896E-02   TIME AVG. MOMENT      7.261E-02
 LARGEST RESIDUAL MOMENT            1.658E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.622E-06   AT NODE        400   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.626E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    11.5     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -17.3     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.649E-05,  FRACTION OF STEP COMPLETED  7.971E-02
 STEP TIME COMPLETED       7.971E-04,  TOTAL TIME COMPLETED        7.971E-04


  INCREMENT    31 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.649E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      5.142E-03   TIME AVG. FORCE       1.644E-02
 LARGEST RESIDUAL FORCE             1.573E-14   AT NODE        497   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.556E-05   AT NODE        386   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        6.401E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.811E-02   TIME AVG. MOMENT      7.215E-02
 LARGEST RESIDUAL MOMENT           -1.545E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.555E-06   AT NODE         70   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.384E-05   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -9.95     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    16.1     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.649E-05,  FRACTION OF STEP COMPLETED  8.636E-02
 STEP TIME COMPLETED       8.636E-04,  TOTAL TIME COMPLETED        8.636E-04


  INCREMENT    32 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.649E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.785E-03   TIME AVG. FORCE       1.608E-02
 LARGEST RESIDUAL FORCE            -1.140E-14   AT NODE        100   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.390E-05   AT NODE        379   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -5.519E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.689E-02   TIME AVG. MOMENT      7.167E-02
 LARGEST RESIDUAL MOMENT            1.676E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.535E-06   AT NODE        371   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.191E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    8.55     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -14.9     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    8.312E-05

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.649E-05,  FRACTION OF STEP COMPLETED  9.301E-02
 STEP TIME COMPLETED       9.301E-04,  TOTAL TIME COMPLETED        9.301E-04


  INCREMENT    33 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.312E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.387E-03   TIME AVG. FORCE       1.572E-02
 LARGEST RESIDUAL FORCE            -1.197E-14   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.313E-05   AT NODE        373   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        6.377E-05   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.207E-02   TIME AVG. MOMENT      7.108E-02
 LARGEST RESIDUAL MOMENT           -1.538E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.010E-06   AT NODE        364   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.505E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    11.3     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    20.7     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.312E-05,  FRACTION OF STEP COMPLETED  0.101    
 STEP TIME COMPLETED       1.013E-03,  TOTAL TIME COMPLETED        1.013E-03


  INCREMENT    34 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.312E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.093E-03   TIME AVG. FORCE       1.538E-02
 LARGEST RESIDUAL FORCE             1.213E-14   AT NODE        490   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.573E-05   AT NODE        347   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.856E-05   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     5.035E-02   TIME AVG. MOMENT      7.047E-02
 LARGEST RESIDUAL MOMENT           -1.027E-14   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.830E-06   AT NODE        357   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.148E-05   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -10.2     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -18.5     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.312E-05,  FRACTION OF STEP COMPLETED  0.110    
 STEP TIME COMPLETED       1.096E-03,  TOTAL TIME COMPLETED        1.096E-03


  INCREMENT    35 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.312E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.763E-03   TIME AVG. FORCE       1.505E-02
 LARGEST RESIDUAL FORCE            -1.114E-14   AT NODE         99   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.650E-05   AT NODE        339   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -4.482E-05   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.847E-02   TIME AVG. MOMENT      6.984E-02
 LARGEST RESIDUAL MOMENT           -1.070E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.627E-06   AT NODE        349   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.705E-06   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    9.14     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    16.5     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.312E-05,  FRACTION OF STEP COMPLETED  0.118    
 STEP TIME COMPLETED       1.179E-03,  TOTAL TIME COMPLETED        1.179E-03


  INCREMENT    36 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.312E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.409E-03   TIME AVG. FORCE       1.473E-02
 LARGEST RESIDUAL FORCE            -1.800E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.034E-05   AT NODE        333   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.328E-05   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.632E-02   TIME AVG. MOMENT      6.918E-02
 LARGEST RESIDUAL MOMENT           -1.223E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.621E-06   AT NODE        341   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.091E-06   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -8.14     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -14.7     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
          TIME INCREMENT MAY NOW INCREASE TO    1.000E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.312E-05,  FRACTION OF STEP COMPLETED  0.126    
 STEP TIME COMPLETED       1.263E-03,  TOTAL TIME COMPLETED        1.263E-03


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


 AVERAGE FORCE                      3.181E-03   TIME AVG. FORCE       1.441E-02
 LARGEST RESIDUAL FORCE            -1.122E-14   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.607E-05   AT NODE        324   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        5.599E-05   AT NODE        329   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.538E-02   TIME AVG. MOMENT      6.854E-02
 LARGEST RESIDUAL MOMENT            1.311E-14   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.668E-06   AT NODE        316   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.148E-05   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    10.2     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    18.4     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.136    
 STEP TIME COMPLETED       1.363E-03,  TOTAL TIME COMPLETED        1.363E-03


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


 AVERAGE FORCE                      2.939E-03   TIME AVG. FORCE       1.411E-02
 LARGEST RESIDUAL FORCE             8.177E-15   AT NODE         99   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.392E-05   AT NODE        317   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        5.531E-05   AT NODE        320   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.409E-02   TIME AVG. MOMENT      6.790E-02
 LARGEST RESIDUAL MOMENT           -1.966E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.546E-06   AT NODE        307   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.963E-06   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -8.91     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -16.0     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.146    
 STEP TIME COMPLETED       1.463E-03,  TOTAL TIME COMPLETED        1.463E-03


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


 AVERAGE FORCE                      2.741E-03   TIME AVG. FORCE       1.382E-02
 LARGEST RESIDUAL FORCE            -1.958E-14   AT NODE        497   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.233E-05   AT NODE        308   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.741E-05   AT NODE        311   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.368E-02   TIME AVG. MOMENT      6.728E-02
 LARGEST RESIDUAL MOMENT            1.171E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.611E-06   AT NODE        298   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     9.508E-06   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    7.71     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    13.8     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.156    
 STEP TIME COMPLETED       1.563E-03,  TOTAL TIME COMPLETED        1.563E-03


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


 AVERAGE FORCE                      2.588E-03   TIME AVG. FORCE       1.354E-02
 LARGEST RESIDUAL FORCE            -1.733E-14   AT NODE        479   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.682E-05   AT NODE        300   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.465E-05   AT NODE        303   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.298E-02   TIME AVG. MOMENT      6.667E-02
 LARGEST RESIDUAL MOMENT           -1.145E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.494E-06   AT NODE        290   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.684E-06   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -6.63     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -11.9     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.166    
 STEP TIME COMPLETED       1.663E-03,  TOTAL TIME COMPLETED        1.663E-03


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


 AVERAGE FORCE                      2.385E-03   TIME AVG. FORCE       1.327E-02
 LARGEST RESIDUAL FORCE            -1.430E-14   AT NODE        484   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.443E-05   AT NODE        271   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.211E-05   AT NODE        294   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.225E-02   TIME AVG. MOMENT      6.607E-02
 LARGEST RESIDUAL MOMENT            9.479E-15   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.467E-06   AT NODE         46   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.973E-06   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    5.67     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    10.2     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.176    
 STEP TIME COMPLETED       1.763E-03,  TOTAL TIME COMPLETED        1.763E-03


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


 AVERAGE FORCE                      2.255E-03   TIME AVG. FORCE       1.300E-02
 LARGEST RESIDUAL FORCE             1.401E-14   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.504E-05   AT NODE        263   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.911E-05   AT NODE        268   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.114E-02   TIME AVG. MOMENT      6.548E-02
 LARGEST RESIDUAL MOMENT           -1.036E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.418E-06   AT NODE        273   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -7.641E-06   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -4.82     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -9.37     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.186    
 STEP TIME COMPLETED       1.863E-03,  TOTAL TIME COMPLETED        1.863E-03


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


 AVERAGE FORCE                      2.187E-03   TIME AVG. FORCE       1.275E-02
 LARGEST RESIDUAL FORCE             1.689E-14   AT NODE         98   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.312E-05   AT NODE        254   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.643E-05   AT NODE        276   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.038E-02   TIME AVG. MOMENT      6.490E-02
 LARGEST RESIDUAL MOMENT           -1.184E-14   AT NODE        489   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.165E-06   AT NODE        265   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.187E-06   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    4.06     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    8.69     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.196    
 STEP TIME COMPLETED       1.963E-03,  TOTAL TIME COMPLETED        1.963E-03


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


 AVERAGE FORCE                      2.081E-03   TIME AVG. FORCE       1.251E-02
 LARGEST RESIDUAL FORCE             1.310E-14   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.668E-05   AT NODE         37   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.391E-05   AT NODE        250   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.931E-02   TIME AVG. MOMENT      6.431E-02
 LARGEST RESIDUAL MOMENT           -1.125E-14   AT NODE        492   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.252E-06   AT NODE        236   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.395E-06   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -3.39     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -8.04     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.206    
 STEP TIME COMPLETED       2.063E-03,  TOTAL TIME COMPLETED        2.063E-03


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


 AVERAGE FORCE                      2.018E-03   TIME AVG. FORCE       1.228E-02
 LARGEST RESIDUAL FORCE             1.348E-14   AT NODE        479   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.692E-05   AT NODE        238   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.763E-05   AT NODE         36   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.937E-02   TIME AVG. MOMENT      6.376E-02
 LARGEST RESIDUAL MOMENT           -1.309E-14   AT NODE         96   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.131E-06   AT NODE        228   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.717E-06   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -3.06     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    7.41     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.216    
 STEP TIME COMPLETED       2.163E-03,  TOTAL TIME COMPLETED        2.163E-03


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


 AVERAGE FORCE                      1.966E-03   TIME AVG. FORCE       1.205E-02
 LARGEST RESIDUAL FORCE            -2.126E-14   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.532E-05   AT NODE        230   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.387E-05   AT NODE        233   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.069E-02   TIME AVG. MOMENT      6.326E-02
 LARGEST RESIDUAL MOMENT           -1.452E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.147E-06   AT NODE        220   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.297E-06   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.81     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -6.81     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.226    
 STEP TIME COMPLETED       2.263E-03,  TOTAL TIME COMPLETED        2.263E-03


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


 AVERAGE FORCE                      1.903E-03   TIME AVG. FORCE       1.184E-02
 LARGEST RESIDUAL FORCE            -2.478E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.005E-05   AT NODE        222   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.162E-05   AT NODE        225   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.043E-02   TIME AVG. MOMENT      6.277E-02
 LARGEST RESIDUAL MOMENT           -2.748E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.095E-06   AT NODE        211   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.811E-06   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -2.57     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    6.24     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.236    
 STEP TIME COMPLETED       2.363E-03,  TOTAL TIME COMPLETED        2.363E-03


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


 AVERAGE FORCE                      1.837E-03   TIME AVG. FORCE       1.163E-02
 LARGEST RESIDUAL FORCE            -2.277E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.117E-05   AT NODE        192   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.904E-05   AT NODE        197   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.100E-02   TIME AVG. MOMENT      6.232E-02
 LARGEST RESIDUAL MOMENT            1.524E-14   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.078E-06   AT NODE        202   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.324E-06   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    2.34     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -5.70     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.246    
 STEP TIME COMPLETED       2.463E-03,  TOTAL TIME COMPLETED        2.463E-03


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


 AVERAGE FORCE                      1.843E-03   TIME AVG. FORCE       1.143E-02
 LARGEST RESIDUAL FORCE            -1.704E-14   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.158E-05   AT NODE        184   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.967E-05   AT NODE        189   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.098E-02   TIME AVG. MOMENT      6.188E-02
 LARGEST RESIDUAL MOMENT           -1.186E-14   AT NODE        489   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.984E-06   AT NODE        194   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.862E-06   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -2.13     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    5.20     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.256    
 STEP TIME COMPLETED       2.563E-03,  TOTAL TIME COMPLETED        2.563E-03


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


 AVERAGE FORCE                      1.750E-03   TIME AVG. FORCE       1.124E-02
 LARGEST RESIDUAL FORCE             2.216E-14   AT NODE        494   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.204E-05   AT NODE        176   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.797E-05   AT NODE        180   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.941E-02   TIME AVG. MOMENT      6.143E-02
 LARGEST RESIDUAL MOMENT            1.687E-14   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.873E-06   AT NODE         17   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -3.416E-06   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.92     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -4.72     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.266    
 STEP TIME COMPLETED       2.663E-03,  TOTAL TIME COMPLETED        2.663E-03


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


 AVERAGE FORCE                      1.703E-03   TIME AVG. FORCE       1.105E-02
 LARGEST RESIDUAL FORCE             2.420E-14   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.068E-05   AT NODE        168   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.746E-05   AT NODE        171   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.058E-02   TIME AVG. MOMENT      6.103E-02
 LARGEST RESIDUAL MOMENT            1.610E-14   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.858E-06   AT NODE         15   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.060E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.73     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    4.28     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.276    
 STEP TIME COMPLETED       2.763E-03,  TOTAL TIME COMPLETED        2.763E-03


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


 AVERAGE FORCE                      1.793E-03   TIME AVG. FORCE       1.087E-02
 LARGEST RESIDUAL FORCE            -2.102E-14   AT NODE        490   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.824E-05   AT NODE        160   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.630E-05   AT NODE        163   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.138E-02   TIME AVG. MOMENT      6.065E-02
 LARGEST RESIDUAL MOMENT           -9.662E-15   AT NODE         97   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.822E-06   AT NODE        149   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.756E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.55     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -3.87     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.286    
 STEP TIME COMPLETED       2.863E-03,  TOTAL TIME COMPLETED        2.863E-03


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


 AVERAGE FORCE                      1.449E-03   TIME AVG. FORCE       1.069E-02
 LARGEST RESIDUAL FORCE            -1.887E-14   AT NODE        489   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.891E-05   AT NODE        127   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.271E-05   AT NODE        136   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.715E-02   TIME AVG. MOMENT      6.020E-02
 LARGEST RESIDUAL MOMENT            1.676E-14   AT NODE        488   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.867E-06   AT NODE        139   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.409E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.38     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    3.50     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.296    
 STEP TIME COMPLETED       2.963E-03,  TOTAL TIME COMPLETED        2.963E-03


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


 AVERAGE FORCE                      1.468E-03   TIME AVG. FORCE       1.052E-02
 LARGEST RESIDUAL FORCE             1.894E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.855E-05   AT NODE        121   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.776E-05   AT NODE        123   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.884E-02   TIME AVG. MOMENT      5.981E-02
 LARGEST RESIDUAL MOMENT           -1.664E-14   AT NODE        458   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.537E-06   AT NODE        133   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.029E-06   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.40     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -3.15     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.306    
 STEP TIME COMPLETED       3.063E-03,  TOTAL TIME COMPLETED        3.063E-03


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


 AVERAGE FORCE                      1.619E-03   TIME AVG. FORCE       1.036E-02
 LARGEST RESIDUAL FORCE            -3.007E-14   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         2.893E-05   AT NODE        180   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -3.916E-05   AT NODE        119   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.204E-02   TIME AVG. MOMENT      5.949E-02
 LARGEST RESIDUAL MOMENT            1.778E-14   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.145E-06   AT NODE        129   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.938E-06   AT NODE        129   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.40     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    2.83     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.316    
 STEP TIME COMPLETED       3.163E-03,  TOTAL TIME COMPLETED        3.163E-03


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


 AVERAGE FORCE                      4.021E-04   TIME AVG. FORCE       1.018E-02
 LARGEST RESIDUAL FORCE             1.752E-14   AT NODE        491   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.742E-05   AT NODE        121   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.752E-05   AT NODE        176   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.727E-02   TIME AVG. MOMENT      5.909E-02
 LARGEST RESIDUAL MOMENT           -1.942E-14   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.601E-06   AT NODE        131   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.644E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.38     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -2.54     AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.326    
 STEP TIME COMPLETED       3.263E-03,  TOTAL TIME COMPLETED        3.263E-03


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


 AVERAGE FORCE                      1.396E-03   TIME AVG. FORCE       1.003E-02
 LARGEST RESIDUAL FORCE             2.992E-14   AT NODE        486   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -6.507E-05   AT NODE          6   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.581E-05   AT NODE        119   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.699E-02   TIME AVG. MOMENT      5.870E-02
 LARGEST RESIDUAL MOMENT           -1.870E-14   AT NODE        488   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.084E-06   AT NODE        112   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -2.227E-06   AT NODE        129   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.35     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -2.32     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.336    
 STEP TIME COMPLETED       3.363E-03,  TOTAL TIME COMPLETED        3.363E-03


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


 AVERAGE FORCE                      1.266E-03   TIME AVG. FORCE       9.877E-03
 LARGEST RESIDUAL FORCE            -2.741E-14   AT NODE        497   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -5.494E-05   AT NODE        132   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.763E-05   AT NODE        173   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.592E-02   TIME AVG. MOMENT      5.831E-02
 LARGEST RESIDUAL MOMENT            2.326E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.492E-06   AT NODE        144   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.358E-06   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.31     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    2.17     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.346    
 STEP TIME COMPLETED       3.463E-03,  TOTAL TIME COMPLETED        3.463E-03


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


 AVERAGE FORCE                      1.266E-03   TIME AVG. FORCE       9.731E-03
 LARGEST RESIDUAL FORCE            -2.539E-14   AT NODE         99   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.257E-05   AT NODE        162   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.635E-05   AT NODE        135   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.872E-02   TIME AVG. MOMENT      5.798E-02
 LARGEST RESIDUAL MOMENT           -2.586E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.794E-06   AT NODE        147   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.096E-06   AT NODE        126   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.27     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -2.02     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.356    
 STEP TIME COMPLETED       3.563E-03,  TOTAL TIME COMPLETED        3.563E-03


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


 AVERAGE FORCE                      1.266E-03   TIME AVG. FORCE       9.590E-03
 LARGEST RESIDUAL FORCE            -2.010E-14   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.828E-05   AT NODE        168   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.482E-05   AT NODE        167   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.908E-02   TIME AVG. MOMENT      5.766E-02
 LARGEST RESIDUAL MOMENT           -1.829E-14   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.655E-06   AT NODE        178   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.436E-06   AT NODE        176   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.21     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.88     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.366    
 STEP TIME COMPLETED       3.663E-03,  TOTAL TIME COMPLETED        3.663E-03


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


 AVERAGE FORCE                      1.324E-03   TIME AVG. FORCE       9.455E-03
 LARGEST RESIDUAL FORCE             1.633E-14   AT NODE         99   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.646E-05   AT NODE        200   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.835E-05   AT NODE        133   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.908E-02   TIME AVG. MOMENT      5.736E-02
 LARGEST RESIDUAL MOMENT            1.871E-14   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.502E-06   AT NODE         23   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.704E-06   AT NODE        125   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.16     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.74     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.376    
 STEP TIME COMPLETED       3.763E-03,  TOTAL TIME COMPLETED        3.763E-03


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


 AVERAGE FORCE                      1.099E-03   TIME AVG. FORCE       9.320E-03
 LARGEST RESIDUAL FORCE            -3.085E-14   AT NODE        492   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.592E-05   AT NODE        141   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.493E-05   AT NODE        181   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.417E-02   TIME AVG. MOMENT      5.698E-02
 LARGEST RESIDUAL MOMENT            2.067E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.828E-06   AT NODE        154   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.625E-06   AT NODE        172   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   -1.10     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.60     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.386    
 STEP TIME COMPLETED       3.863E-03,  TOTAL TIME COMPLETED        3.863E-03


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


 AVERAGE FORCE                      1.133E-03   TIME AVG. FORCE       9.190E-03
 LARGEST RESIDUAL FORCE             2.449E-14   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.375E-05   AT NODE         19   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.987E-05   AT NODE        117   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.247E-02   TIME AVG. MOMENT      5.659E-02
 LARGEST RESIDUAL MOMENT           -2.013E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.719E-06   AT NODE        185   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.243E-06   AT NODE        202   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS    1.03     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.47     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.396    
 STEP TIME COMPLETED       3.963E-03,  TOTAL TIME COMPLETED        3.963E-03


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


 AVERAGE FORCE                      1.170E-03   TIME AVG. FORCE       9.065E-03
 LARGEST RESIDUAL FORCE             2.973E-14   AT NODE         99   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.810E-05   AT NODE        132   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        2.459E-05   AT NODE        177   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.452E-02   TIME AVG. MOMENT      5.625E-02
 LARGEST RESIDUAL MOMENT           -1.389E-14   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.640E-06   AT NODE        191   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.389E-06   AT NODE        169   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.972     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.35     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.406    
 STEP TIME COMPLETED       4.063E-03,  TOTAL TIME COMPLETED        4.063E-03


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


 AVERAGE FORCE                      1.136E-03   TIME AVG. FORCE       8.943E-03
 LARGEST RESIDUAL FORCE             2.230E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.169E-05   AT NODE        186   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.243E-05   AT NODE        206   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.574E-02   TIME AVG. MOMENT      5.593E-02
 LARGEST RESIDUAL MOMENT           -1.733E-14   AT NODE         96   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.916E-06   AT NODE        198   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.272E-06   AT NODE         25   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.910     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.23     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.416    
 STEP TIME COMPLETED       4.163E-03,  TOTAL TIME COMPLETED        4.163E-03


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


 AVERAGE FORCE                      1.326E-03   TIME AVG. FORCE       8.827E-03
 LARGEST RESIDUAL FORCE            -2.178E-14   AT NODE        477   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.612E-05   AT NODE        219   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.998E-05   AT NODE        234   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.751E-02   TIME AVG. MOMENT      5.565E-02
 LARGEST RESIDUAL MOMENT            2.527E-14   AT NODE        491   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.690E-06   AT NODE        207   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.271E-06   AT NODE        226   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.848     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS    1.12     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.426    
 STEP TIME COMPLETED       4.263E-03,  TOTAL TIME COMPLETED        4.263E-03


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


 AVERAGE FORCE                      1.257E-03   TIME AVG. FORCE       8.714E-03
 LARGEST RESIDUAL FORCE            -2.873E-14   AT NODE        497   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.968E-05   AT NODE        227   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.882E-05   AT NODE        245   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.955E-02   TIME AVG. MOMENT      5.541E-02
 LARGEST RESIDUAL MOMENT            1.875E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.735E-06   AT NODE        238   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     9.234E-07   AT NODE        237   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.788     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   -1.02     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.436    
 STEP TIME COMPLETED       4.363E-03,  TOTAL TIME COMPLETED        4.363E-03


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


 AVERAGE FORCE                      1.294E-03   TIME AVG. FORCE       8.605E-03
 LARGEST RESIDUAL FORCE            -3.784E-14   AT NODE        482   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.775E-05   AT NODE        233   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.090E-05   AT NODE        230   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.050E-02   TIME AVG. MOMENT      5.519E-02
 LARGEST RESIDUAL MOMENT            1.786E-14   AT NODE         96   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.719E-06   AT NODE         37   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.162E-06   AT NODE        222   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.730     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.925     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.446    
 STEP TIME COMPLETED       4.463E-03,  TOTAL TIME COMPLETED        4.463E-03


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


 AVERAGE FORCE                      1.365E-03   TIME AVG. FORCE       8.500E-03
 LARGEST RESIDUAL FORCE             2.982E-14   AT NODE        481   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.153E-05   AT NODE        241   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.911E-05   AT NODE        241   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.931E-02   TIME AVG. MOMENT      5.496E-02
 LARGEST RESIDUAL MOMENT            2.589E-14   AT NODE         96   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.644E-06   AT NODE         39   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.038E-06   AT NODE        249   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.674     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.836     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.456    
 STEP TIME COMPLETED       4.563E-03,  TOTAL TIME COMPLETED        4.563E-03


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


 AVERAGE FORCE                      1.260E-03   TIME AVG. FORCE       8.397E-03
 LARGEST RESIDUAL FORCE            -3.035E-14   AT NODE        486   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.194E-05   AT NODE        275   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.846E-05   AT NODE        268   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.881E-02   TIME AVG. MOMENT      5.473E-02
 LARGEST RESIDUAL MOMENT           -2.031E-14   AT NODE        482   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.648E-06   AT NODE         41   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.039E-06   AT NODE        260   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.620     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.753     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.466    
 STEP TIME COMPLETED       4.663E-03,  TOTAL TIME COMPLETED        4.663E-03


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


 AVERAGE FORCE                      1.150E-03   TIME AVG. FORCE       8.295E-03
 LARGEST RESIDUAL FORCE            -3.722E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.445E-05   AT NODE         46   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.855E-05   AT NODE         45   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.755E-02   TIME AVG. MOMENT      5.449E-02
 LARGEST RESIDUAL MOMENT            2.207E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.550E-06   AT NODE         43   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.945E-07   AT NODE         43   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.569     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.676     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.476    
 STEP TIME COMPLETED       4.763E-03,  TOTAL TIME COMPLETED        4.763E-03


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


 AVERAGE FORCE                      1.102E-03   TIME AVG. FORCE       8.195E-03
 LARGEST RESIDUAL FORCE            -2.969E-14   AT NODE         95   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.529E-05   AT NODE         48   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.605E-05   AT NODE        287   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.376E-02   TIME AVG. MOMENT      5.420E-02
 LARGEST RESIDUAL MOMENT           -1.744E-14   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.436E-06   AT NODE         51   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.862E-07   AT NODE        296   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.521     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.605     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.486    
 STEP TIME COMPLETED       4.863E-03,  TOTAL TIME COMPLETED        4.863E-03


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


 AVERAGE FORCE                      1.109E-03   TIME AVG. FORCE       8.098E-03
 LARGEST RESIDUAL FORCE            -2.447E-14   AT NODE        497   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.515E-05   AT NODE        297   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.562E-05   AT NODE        295   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.139E-02   TIME AVG. MOMENT      5.389E-02
 LARGEST RESIDUAL MOMENT           -2.103E-14   AT NODE        469   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.456E-06   AT NODE        309   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.446E-07   AT NODE         52   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.475     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.540     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.496    
 STEP TIME COMPLETED       4.963E-03,  TOTAL TIME COMPLETED        4.963E-03


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


 AVERAGE FORCE                      1.186E-03   TIME AVG. FORCE       8.004E-03
 LARGEST RESIDUAL FORCE             2.838E-14   AT NODE        488   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.169E-05   AT NODE        305   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.640E-05   AT NODE        323   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.020E-02   TIME AVG. MOMENT      5.357E-02
 LARGEST RESIDUAL MOMENT           -2.059E-14   AT NODE         98   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.472E-06   AT NODE         55   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.795E-07   AT NODE        314   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.432     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.481     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.506    
 STEP TIME COMPLETED       5.063E-03,  TOTAL TIME COMPLETED        5.063E-03


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


 AVERAGE FORCE                      1.223E-03   TIME AVG. FORCE       7.914E-03
 LARGEST RESIDUAL FORCE            -2.278E-14   AT NODE        484   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.907E-05   AT NODE        312   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.695E-05   AT NODE        332   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.090E-02   TIME AVG. MOMENT      5.327E-02
 LARGEST RESIDUAL MOMENT           -1.565E-14   AT NODE        488   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.542E-06   AT NODE        325   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.249E-07   AT NODE        323   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.392     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.426     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.516    
 STEP TIME COMPLETED       5.163E-03,  TOTAL TIME COMPLETED        5.163E-03


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


 AVERAGE FORCE                      1.208E-03   TIME AVG. FORCE       7.826E-03
 LARGEST RESIDUAL FORCE             3.990E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.575E-05   AT NODE        319   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.592E-05   AT NODE        341   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.267E-02   TIME AVG. MOMENT      5.300E-02
 LARGEST RESIDUAL MOMENT            2.141E-14   AT NODE        469   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.455E-06   AT NODE        333   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -7.680E-07   AT NODE        330   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.355     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.377     AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.526    
 STEP TIME COMPLETED       5.263E-03,  TOTAL TIME COMPLETED        5.263E-03


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


 AVERAGE FORCE                      1.206E-03   TIME AVG. FORCE       7.740E-03
 LARGEST RESIDUAL FORCE            -2.584E-14   AT NODE        489   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.243E-05   AT NODE        209   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.538E-05   AT NODE        349   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.441E-02   TIME AVG. MOMENT      5.276E-02
 LARGEST RESIDUAL MOMENT            1.908E-14   AT NODE        488   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.372E-06   AT NODE        341   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.162E-07   AT NODE        358   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.321     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.336     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.536    
 STEP TIME COMPLETED       5.363E-03,  TOTAL TIME COMPLETED        5.363E-03


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


 AVERAGE FORCE                      1.283E-03   TIME AVG. FORCE       7.657E-03
 LARGEST RESIDUAL FORCE            -2.720E-14   AT NODE        478   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.141E-05   AT NODE        360   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.523E-05   AT NODE         65   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.731E-02   TIME AVG. MOMENT      5.256E-02
 LARGEST RESIDUAL MOMENT           -1.938E-14   AT NODE        479   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.228E-06   AT NODE        349   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.900E-07   AT NODE        366   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.289     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.309     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.546    
 STEP TIME COMPLETED       5.463E-03,  TOTAL TIME COMPLETED        5.463E-03


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


 AVERAGE FORCE                      1.374E-03   TIME AVG. FORCE       7.577E-03
 LARGEST RESIDUAL FORCE            -2.660E-14   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.246E-05   AT NODE        369   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.400E-05   AT NODE         67   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.986E-02   TIME AVG. MOMENT      5.240E-02
 LARGEST RESIDUAL MOMENT            1.577E-14   AT NODE         97   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.160E-06   AT NODE        380   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.335E-07   AT NODE        374   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.259     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.283     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.556    
 STEP TIME COMPLETED       5.563E-03,  TOTAL TIME COMPLETED        5.563E-03


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


 AVERAGE FORCE                      1.345E-03   TIME AVG. FORCE       7.500E-03
 LARGEST RESIDUAL FORCE            -2.861E-14   AT NODE        478   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.556E-05   AT NODE        376   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.424E-05   AT NODE        374   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.971E-02   TIME AVG. MOMENT      5.224E-02
 LARGEST RESIDUAL MOMENT            1.827E-14   AT NODE         92   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.224E-06   AT NODE        389   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.017E-07   AT NODE        383   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.232     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.259     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.566    
 STEP TIME COMPLETED       5.663E-03,  TOTAL TIME COMPLETED        5.663E-03


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


 AVERAGE FORCE                      1.334E-03   TIME AVG. FORCE       7.423E-03
 LARGEST RESIDUAL FORCE             2.670E-14   AT NODE        480   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.541E-05   AT NODE        432   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.439E-05   AT NODE        404   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.021E-02   TIME AVG. MOMENT      5.209E-02
 LARGEST RESIDUAL MOMENT            2.964E-14   AT NODE        463   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.269E-06   AT NODE        419   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     7.287E-07   AT NODE        392   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.207     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.236     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.576    
 STEP TIME COMPLETED       5.763E-03,  TOTAL TIME COMPLETED        5.763E-03


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


 AVERAGE FORCE                      1.382E-03   TIME AVG. FORCE       7.350E-03
 LARGEST RESIDUAL FORCE             3.599E-14   AT NODE        497   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.568E-05   AT NODE        441   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.469E-05   AT NODE        412   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.206E-02   TIME AVG. MOMENT      5.197E-02
 LARGEST RESIDUAL MOMENT           -2.704E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.225E-06   AT NODE        427   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.990E-07   AT NODE        424   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.185     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.214     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.586    
 STEP TIME COMPLETED       5.863E-03,  TOTAL TIME COMPLETED        5.863E-03


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


 AVERAGE FORCE                      1.336E-03   TIME AVG. FORCE       7.277E-03
 LARGEST RESIDUAL FORCE            -3.458E-14   AT NODE        492   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.861E-05   AT NODE        448   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.413E-05   AT NODE        419   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.991E-02   TIME AVG. MOMENT      5.182E-02
 LARGEST RESIDUAL MOMENT           -1.914E-14   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.230E-06   AT NODE        435   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.884E-07   AT NODE        430   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.164     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.194     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.596    
 STEP TIME COMPLETED       5.963E-03,  TOTAL TIME COMPLETED        5.963E-03


  INCREMENT    84 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.850E-04   TIME AVG. FORCE       7.195E-03
 LARGEST RESIDUAL FORCE            -3.375E-14   AT NODE        462   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.452E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.373E-05   AT NODE        428   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.855E-02   TIME AVG. MOMENT      5.167E-02
 LARGEST RESIDUAL MOMENT            2.081E-14   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.412E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.754E-07   AT NODE        419   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.145     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.175     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.606    
 STEP TIME COMPLETED       6.063E-03,  TOTAL TIME COMPLETED        6.063E-03


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


 AVERAGE FORCE                      1.361E-03   TIME AVG. FORCE       7.127E-03
 LARGEST RESIDUAL FORCE             2.941E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.912E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.780E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.353E-02   TIME AVG. MOMENT      5.157E-02
 LARGEST RESIDUAL MOMENT           -2.479E-14   AT NODE        489   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.397E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.003E-07   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   0.128     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.158     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.616    
 STEP TIME COMPLETED       6.163E-03,  TOTAL TIME COMPLETED        6.163E-03


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


 AVERAGE FORCE                      1.227E-03   TIME AVG. FORCE       7.058E-03
 LARGEST RESIDUAL FORCE            -2.635E-14   AT NODE        460   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.125E-05   AT NODE        473   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.385E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.182E-02   TIME AVG. MOMENT      5.146E-02
 LARGEST RESIDUAL MOMENT           -2.126E-14   AT NODE         90   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      8.265E-07   AT NODE        460   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.051E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -0.113     AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.142     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.626    
 STEP TIME COMPLETED       6.263E-03,  TOTAL TIME COMPLETED        6.263E-03


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


 AVERAGE FORCE                      2.554E-04   TIME AVG. FORCE       6.980E-03
 LARGEST RESIDUAL FORCE             2.824E-14   AT NODE         90   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.952E-05   AT NODE        481   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.037E-05   AT NODE        454   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.175E-02   TIME AVG. MOMENT      5.123E-02
 LARGEST RESIDUAL MOMENT            1.671E-14   AT NODE        489   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.469E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.994E-07   AT NODE        445   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   9.908E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.128     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.636    
 STEP TIME COMPLETED       6.363E-03,  TOTAL TIME COMPLETED        6.363E-03


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


 AVERAGE FORCE                      1.039E-03   TIME AVG. FORCE       6.912E-03
 LARGEST RESIDUAL FORCE             3.855E-14   AT NODE        479   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.838E-05   AT NODE        482   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        3.062E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.664E-02   TIME AVG. MOMENT      5.106E-02
 LARGEST RESIDUAL MOMENT            2.979E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -2.609E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     1.591E-06   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -8.660E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   0.114     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.646    
 STEP TIME COMPLETED       6.463E-03,  TOTAL TIME COMPLETED        6.463E-03


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


 AVERAGE FORCE                      1.047E-03   TIME AVG. FORCE       6.846E-03
 LARGEST RESIDUAL FORCE             2.660E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.830E-05   AT NODE        422   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.745E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.755E-02   TIME AVG. MOMENT      5.091E-02
 LARGEST RESIDUAL MOMENT           -2.759E-14   AT NODE        485   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.022E-06   AT NODE         78   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     9.936E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   7.597E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -0.102     AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.656    
 STEP TIME COMPLETED       6.563E-03,  TOTAL TIME COMPLETED        6.563E-03


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


 AVERAGE FORCE                      9.711E-04   TIME AVG. FORCE       6.781E-03
 LARGEST RESIDUAL FORCE            -2.946E-14   AT NODE         91   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.979E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.543E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.411E-02   TIME AVG. MOMENT      5.073E-02
 LARGEST RESIDUAL MOMENT           -1.895E-14   AT NODE        490   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.231E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -9.182E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -6.568E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   9.072E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.666    
 STEP TIME COMPLETED       6.663E-03,  TOTAL TIME COMPLETED        6.663E-03


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


 AVERAGE FORCE                      3.462E-04   TIME AVG. FORCE       6.710E-03
 LARGEST RESIDUAL FORCE             3.186E-14   AT NODE        458   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         6.689E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -2.762E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.239E-02   TIME AVG. MOMENT      5.052E-02
 LARGEST RESIDUAL MOMENT           -2.642E-14   AT NODE         90   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      2.313E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -1.497E-06   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   5.649E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -8.082E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.676    
 STEP TIME COMPLETED       6.763E-03,  TOTAL TIME COMPLETED        6.763E-03


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


 AVERAGE FORCE                      3.115E-04   TIME AVG. FORCE       6.641E-03
 LARGEST RESIDUAL FORCE            -3.988E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         5.431E-05   AT NODE        447   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.250E-05   AT NODE        448   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.079E-02   TIME AVG. MOMENT      5.031E-02
 LARGEST RESIDUAL MOMENT            1.825E-14   AT NODE        489   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.669E-06   AT NODE        459   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.175E-07   AT NODE        437   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.899E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   7.149E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.686    
 STEP TIME COMPLETED       6.863E-03,  TOTAL TIME COMPLETED        6.863E-03


  INCREMENT    93 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.044E-03   TIME AVG. FORCE       6.581E-03
 LARGEST RESIDUAL FORCE             2.546E-14   AT NODE        464   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.566E-05   AT NODE        444   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.674E-05   AT NODE        469   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.460E-02   TIME AVG. MOMENT      5.014E-02
 LARGEST RESIDUAL MOMENT            2.279E-14   AT NODE        487   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.319E-06   AT NODE        428   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     9.553E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   4.208E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -6.320E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.696    
 STEP TIME COMPLETED       6.963E-03,  TOTAL TIME COMPLETED        6.963E-03


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


 AVERAGE FORCE                      1.100E-03   TIME AVG. FORCE       6.522E-03
 LARGEST RESIDUAL FORCE            -2.880E-14   AT NODE        484   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.418E-05   AT NODE        381   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.342E-05   AT NODE        412   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.559E-02   TIME AVG. MOMENT      4.999E-02
 LARGEST RESIDUAL MOMENT            2.169E-14   AT NODE         96   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.264E-06   AT NODE        422   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.376E-07   AT NODE        422   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -3.556E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   5.602E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.706    
 STEP TIME COMPLETED       7.063E-03,  TOTAL TIME COMPLETED        7.063E-03


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


 AVERAGE FORCE                      1.091E-03   TIME AVG. FORCE       6.465E-03
 LARGEST RESIDUAL FORCE            -4.299E-14   AT NODE        463   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.643E-05   AT NODE        376   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.227E-05   AT NODE         83   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.816E-02   TIME AVG. MOMENT      4.986E-02
 LARGEST RESIDUAL MOMENT            2.457E-14   AT NODE        465   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.000E-06   AT NODE        362   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -8.019E-07   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   3.001E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -4.944E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.716    
 STEP TIME COMPLETED       7.163E-03,  TOTAL TIME COMPLETED        7.163E-03


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


 AVERAGE FORCE                      9.454E-04   TIME AVG. FORCE       6.408E-03
 LARGEST RESIDUAL FORCE             2.579E-14   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.417E-05   AT NODE         94   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.140E-05   AT NODE        397   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.466E-02   TIME AVG. MOMENT      4.970E-02
 LARGEST RESIDUAL MOMENT           -2.171E-14   AT NODE        458   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.232E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.025E-07   AT NODE        408   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -2.556E-02 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   4.328E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.726    
 STEP TIME COMPLETED       7.263E-03,  TOTAL TIME COMPLETED        7.263E-03


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


 AVERAGE FORCE                      9.947E-04   TIME AVG. FORCE       6.352E-03
 LARGEST RESIDUAL FORCE             2.500E-14   AT NODE        478   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.952E-05   AT NODE        408   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.439E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.101E-02   TIME AVG. MOMENT      4.951E-02
 LARGEST RESIDUAL MOMENT            1.977E-14   AT NODE        479   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.837E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     8.412E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   2.222E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -3.795E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.736    
 STEP TIME COMPLETED       7.363E-03,  TOTAL TIME COMPLETED        7.363E-03


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


 AVERAGE FORCE                      1.007E-03   TIME AVG. FORCE       6.297E-03
 LARGEST RESIDUAL FORCE            -3.647E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.657E-05   AT NODE        401   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.141E-05   AT NODE        402   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.282E-02   TIME AVG. MOMENT      4.934E-02
 LARGEST RESIDUAL MOMENT           -2.873E-14   AT NODE        451   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.277E-06   AT NODE        388   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.188E-07   AT NODE        392   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -2.026E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   3.338E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.746    
 STEP TIME COMPLETED       7.463E-03,  TOTAL TIME COMPLETED        7.463E-03


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


 AVERAGE FORCE                      1.096E-03   TIME AVG. FORCE       6.245E-03
 LARGEST RESIDUAL FORCE            -3.616E-14   AT NODE        467   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.515E-05   AT NODE        395   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.080E-05   AT NODE        372   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.598E-02   TIME AVG. MOMENT      4.920E-02
 LARGEST RESIDUAL MOMENT           -1.730E-14   AT NODE         98   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.493E-06   AT NODE        380   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.139E-07   AT NODE        409   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.856E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -2.918E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.756    
 STEP TIME COMPLETED       7.563E-03,  TOTAL TIME COMPLETED        7.563E-03


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


 AVERAGE FORCE                      1.206E-03   TIME AVG. FORCE       6.194E-03
 LARGEST RESIDUAL FORCE            -2.448E-14   AT NODE        489   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.540E-05   AT NODE        391   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.248E-05   AT NODE        367   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.022E-02   TIME AVG. MOMENT      4.911E-02
 LARGEST RESIDUAL MOMENT           -2.104E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.301E-06   AT NODE        375   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.704E-07   AT NODE        377   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.706E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   2.529E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.766    
 STEP TIME COMPLETED       7.663E-03,  TOTAL TIME COMPLETED        7.663E-03


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


 AVERAGE FORCE                      1.242E-03   TIME AVG. FORCE       6.145E-03
 LARGEST RESIDUAL FORCE            -2.606E-14   AT NODE        489   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.112E-05   AT NODE        328   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.142E-05   AT NODE         59   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.466E-02   TIME AVG. MOMENT      4.907E-02
 LARGEST RESIDUAL MOMENT           -1.654E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.323E-06   AT NODE         61   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.905E-07   AT NODE        345   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.559E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   2.244E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.776    
 STEP TIME COMPLETED       7.763E-03,  TOTAL TIME COMPLETED        7.763E-03


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


 AVERAGE FORCE                      1.232E-03   TIME AVG. FORCE       6.097E-03
 LARGEST RESIDUAL FORCE            -2.767E-14   AT NODE         96   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.986E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.081E-05   AT NODE        350   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.197E-02   TIME AVG. MOMENT      4.900E-02
 LARGEST RESIDUAL MOMENT           -2.117E-14   AT NODE        458   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.169E-06   AT NODE        334   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.722E-07   AT NODE         66   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.403E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -2.054E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.786    
 STEP TIME COMPLETED       7.863E-03,  TOTAL TIME COMPLETED        7.863E-03


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


 AVERAGE FORCE                      1.163E-03   TIME AVG. FORCE       6.049E-03
 LARGEST RESIDUAL FORCE            -7.298E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.801E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.070E-05   AT NODE        319   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.806E-02   TIME AVG. MOMENT      4.889E-02
 LARGEST RESIDUAL MOMENT           -3.867E-14   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.203E-06   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.771E-07   AT NODE        330   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.262E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.877E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.796    
 STEP TIME COMPLETED       7.963E-03,  TOTAL TIME COMPLETED        7.963E-03


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


 AVERAGE FORCE                      1.071E-03   TIME AVG. FORCE       6.001E-03
 LARGEST RESIDUAL FORCE            -6.029E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.537E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.064E-05   AT NODE         53   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.707E-02   TIME AVG. MOMENT      4.878E-02
 LARGEST RESIDUAL MOMENT           -3.509E-14   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.162E-06   AT NODE        294   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.193E-07   AT NODE         56   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.145E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.724E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.806    
 STEP TIME COMPLETED       8.063E-03,  TOTAL TIME COMPLETED        8.063E-03


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


 AVERAGE FORCE                      1.008E-03   TIME AVG. FORCE       5.954E-03
 LARGEST RESIDUAL FORCE             5.688E-14   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         4.239E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        9.824E-06   AT NODE        325   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.671E-02   TIME AVG. MOMENT      4.867E-02
 LARGEST RESIDUAL MOMENT            4.890E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.191E-06   AT NODE        286   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.982E-07   AT NODE        314   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.033E-02 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.578E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.816    
 STEP TIME COMPLETED       8.163E-03,  TOTAL TIME COMPLETED        8.163E-03


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


 AVERAGE FORCE                      1.078E-03   TIME AVG. FORCE       5.908E-03
 LARGEST RESIDUAL FORCE             3.962E-14   AT NODE        497   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -4.157E-05   AT NODE         42   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.046E-05   AT NODE        295   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.077E-02   TIME AVG. MOMENT      4.859E-02
 LARGEST RESIDUAL MOMENT            4.117E-14   AT NODE        100   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.120E-06   AT NODE        278   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.365E-07   AT NODE         52   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -9.206E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.431E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.826    
 STEP TIME COMPLETED       8.263E-03,  TOTAL TIME COMPLETED        8.263E-03


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


 AVERAGE FORCE                      1.179E-03   TIME AVG. FORCE       5.864E-03
 LARGEST RESIDUAL FORCE            -7.132E-14   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.634E-05   AT NODE        259   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.788E-06   AT NODE        286   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.320E-02   TIME AVG. MOMENT      4.854E-02
 LARGEST RESIDUAL MOMENT            3.733E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -9.974E-07   AT NODE        245   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.182E-07   AT NODE        275   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   8.140E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.289E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.836    
 STEP TIME COMPLETED       8.363E-03,  TOTAL TIME COMPLETED        8.363E-03


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


 AVERAGE FORCE                      1.271E-03   TIME AVG. FORCE       5.821E-03
 LARGEST RESIDUAL FORCE             4.054E-14   AT NODE        494   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.053E-05   AT NODE        251   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -8.725E-06   AT NODE         45   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.203E-02   TIME AVG. MOMENT      4.848E-02
 LARGEST RESIDUAL MOMENT            3.013E-14   AT NODE        495   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.034E-06   AT NODE         35   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.317E-07   AT NODE        290   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -7.225E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -1.164E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.846    
 STEP TIME COMPLETED       8.463E-03,  TOTAL TIME COMPLETED        8.463E-03


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


 AVERAGE FORCE                      1.232E-03   TIME AVG. FORCE       5.779E-03
 LARGEST RESIDUAL FORCE             6.285E-14   AT NODE        496   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.044E-05   AT NODE        291   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.958E-06   AT NODE         43   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.120E-02   TIME AVG. MOMENT      4.841E-02
 LARGEST RESIDUAL MOMENT            4.563E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -9.295E-07   AT NODE        231   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.816E-07   AT NODE        259   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   6.417E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   1.054E-02 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.856    
 STEP TIME COMPLETED       8.563E-03,  TOTAL TIME COMPLETED        8.563E-03


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


 AVERAGE FORCE                      1.141E-03   TIME AVG. FORCE       5.737E-03
 LARGEST RESIDUAL FORCE             5.215E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.040E-05   AT NODE        286   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        9.515E-06   AT NODE        239   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.193E-02   TIME AVG. MOMENT      4.836E-02
 LARGEST RESIDUAL MOMENT            3.078E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -8.676E-07   AT NODE        223   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.767E-07   AT NODE        250   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -5.703E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -9.518E-03 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.866    
 STEP TIME COMPLETED       8.663E-03,  TOTAL TIME COMPLETED        8.663E-03


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


 AVERAGE FORCE                      1.074E-03   TIME AVG. FORCE       5.695E-03
 LARGEST RESIDUAL FORCE            -3.809E-14   AT NODE        495   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         2.989E-05   AT NODE        467   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.014E-05   AT NODE        230   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.055E-02   TIME AVG. MOMENT      4.828E-02
 LARGEST RESIDUAL MOMENT           -3.506E-14   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -9.332E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.727E-07   AT NODE         36   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   4.971E-03 AND OCCURS AT NODE        499 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   8.491E-03 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.876    
 STEP TIME COMPLETED       8.763E-03,  TOTAL TIME COMPLETED        8.763E-03


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


 AVERAGE FORCE                      1.086E-03   TIME AVG. FORCE       5.654E-03
 LARGEST RESIDUAL FORCE             7.302E-14   AT NODE        494   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.530E-05   AT NODE        468   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        9.426E-06   AT NODE        224   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.779E-02   TIME AVG. MOMENT      4.819E-02
 LARGEST RESIDUAL MOMENT            4.562E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -8.708E-07   AT NODE        101   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.961E-07   AT NODE        235   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   4.634E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -7.509E-03 AND OCCURS AT NODE        100 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.886    
 STEP TIME COMPLETED       8.863E-03,  TOTAL TIME COMPLETED        8.863E-03


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


 AVERAGE FORCE                      1.037E-03   TIME AVG. FORCE       5.613E-03
 LARGEST RESIDUAL FORCE            -5.489E-14   AT NODE        500   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.706E-05   AT NODE        469   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        8.248E-06   AT NODE        215   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.409E-02   TIME AVG. MOMENT      4.807E-02
 LARGEST RESIDUAL MOMENT           -4.133E-14   AT NODE        501   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      8.600E-07   AT NODE        223   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -4.200E-07   AT NODE        227   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -4.307E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -6.912E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.896    
 STEP TIME COMPLETED       8.963E-03,  TOTAL TIME COMPLETED        8.963E-03


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


 AVERAGE FORCE                      9.886E-04   TIME AVG. FORCE       5.572E-03
 LARGEST RESIDUAL FORCE             4.553E-14   AT NODE        493   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.433E-05   AT NODE        466   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        9.604E-06   AT NODE        206   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.289E-02   TIME AVG. MOMENT      4.793E-02
 LARGEST RESIDUAL MOMENT            2.343E-14   AT NODE         98   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      9.604E-07   AT NODE        216   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.665E-07   AT NODE        195   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   3.935E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   6.315E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.906    
 STEP TIME COMPLETED       9.063E-03,  TOTAL TIME COMPLETED        9.063E-03


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


 AVERAGE FORCE                      1.100E-03   TIME AVG. FORCE       5.533E-03
 LARGEST RESIDUAL FORCE             6.227E-14   AT NODE        501   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.345E-05   AT NODE        369   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.316E-06   AT NODE        176   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.540E-02   TIME AVG. MOMENT      4.782E-02
 LARGEST RESIDUAL MOMENT           -3.856E-14   AT NODE        500   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -9.077E-07   AT NODE        134   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.639E-07   AT NODE        188   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -3.621E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -5.818E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.916    
 STEP TIME COMPLETED       9.163E-03,  TOTAL TIME COMPLETED        9.163E-03


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


 AVERAGE FORCE                      1.121E-03   TIME AVG. FORCE       5.495E-03
 LARGEST RESIDUAL FORCE            -3.626E-14   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.410E-05   AT NODE        368   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -9.229E-06   AT NODE        168   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.670E-02   TIME AVG. MOMENT      4.773E-02
 LARGEST RESIDUAL MOMENT            3.734E-14   AT NODE        496   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -8.015E-07   AT NODE        176   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     4.536E-07   AT NODE        179   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   3.383E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   5.437E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.926    
 STEP TIME COMPLETED       9.263E-03,  TOTAL TIME COMPLETED        9.263E-03


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


 AVERAGE FORCE                      3.024E-04   TIME AVG. FORCE       5.451E-03
 LARGEST RESIDUAL FORCE            -3.551E-14   AT NODE        494   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.560E-05   AT NODE        180   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        7.704E-06   AT NODE        183   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.529E-02   TIME AVG. MOMENT      4.762E-02
 LARGEST RESIDUAL MOMENT            3.197E-14   AT NODE         99   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -9.496E-07   AT NODE        166   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.852E-07   AT NODE        172   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -3.113E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -5.037E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.936    
 STEP TIME COMPLETED       9.363E-03,  TOTAL TIME COMPLETED        9.363E-03


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


 AVERAGE FORCE                      1.062E-03   TIME AVG. FORCE       5.414E-03
 LARGEST RESIDUAL FORCE            -5.070E-14   AT NODE        490   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.758E-05   AT NODE        175   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.217E-05   AT NODE        124   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.027E-02   TIME AVG. MOMENT      4.756E-02
 LARGEST RESIDUAL MOMENT           -2.269E-14   AT NODE        499   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      1.172E-06   AT NODE        136   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -5.423E-07   AT NODE        137   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   2.819E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   4.584E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.946    
 STEP TIME COMPLETED       9.463E-03,  TOTAL TIME COMPLETED        9.463E-03


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


 AVERAGE FORCE                      1.307E-03   TIME AVG. FORCE       5.379E-03
 LARGEST RESIDUAL FORCE             4.754E-14   AT NODE        498   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -2.974E-05   AT NODE        171   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.408E-05   AT NODE          6   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.568E-02   TIME AVG. MOMENT      4.754E-02
 LARGEST RESIDUAL MOMENT           -3.762E-14   AT NODE        493   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -6.638E-07   AT NODE        157   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION    -6.281E-07   AT NODE        134   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -2.544E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -4.155E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.956    
 STEP TIME COMPLETED       9.563E-03,  TOTAL TIME COMPLETED        9.563E-03


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


 AVERAGE FORCE                      1.099E-03   TIME AVG. FORCE       5.344E-03
 LARGEST RESIDUAL FORCE            -4.391E-14   AT NODE        490   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         2.796E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        4.934E-06   AT NODE        234   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     4.254E-02   TIME AVG. MOMENT      4.750E-02
 LARGEST RESIDUAL MOMENT            4.122E-14   AT NODE        497   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -8.407E-07   AT NODE         10   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     2.537E-07   AT NODE        224   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   2.320E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   3.803E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.966    
 STEP TIME COMPLETED       9.663E-03,  TOTAL TIME COMPLETED        9.663E-03


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


 AVERAGE FORCE                      3.104E-04   TIME AVG. FORCE       5.302E-03
 LARGEST RESIDUAL FORCE             6.376E-14   AT NODE         99   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.914E-05   AT NODE        151   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.369E-05   AT NODE        122   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.710E-02   TIME AVG. MOMENT      4.742E-02
 LARGEST RESIDUAL MOMENT            3.726E-14   AT NODE        494   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.613E-06   AT NODE        136   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.850E-07   AT NODE        134   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -2.107E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -3.494E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.976    
 STEP TIME COMPLETED       9.763E-03,  TOTAL TIME COMPLETED        9.763E-03


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


 AVERAGE FORCE                      1.082E-03   TIME AVG. FORCE       5.267E-03
 LARGEST RESIDUAL FORCE             4.174E-14   AT NODE        100   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.        -3.879E-05   AT NODE        156   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -1.675E-05   AT NODE          6   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.770E-02   TIME AVG. MOMENT      4.734E-02
 LARGEST RESIDUAL MOMENT            3.502E-14   AT NODE        498   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION     -1.240E-06   AT NODE         10   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     6.662E-07   AT NODE          9   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   1.900E-03 AND OCCURS AT NODE        498 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   3.195E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.986    
 STEP TIME COMPLETED       9.863E-03,  TOTAL TIME COMPLETED        9.863E-03


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


 AVERAGE FORCE                      1.087E-03   TIME AVG. FORCE       5.233E-03
 LARGEST RESIDUAL FORCE             4.525E-14   AT NODE        499   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         3.812E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.       -8.896E-06   AT NODE        186   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.499E-02   TIME AVG. MOMENT      4.724E-02
 LARGEST RESIDUAL MOMENT            3.432E-14   AT NODE        489   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      9.492E-07   AT NODE         19   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     3.952E-07   AT NODE        197   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS  -1.814E-03 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS  -2.900E-03 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  0.996    
 STEP TIME COMPLETED       9.963E-03,  TOTAL TIME COMPLETED        9.963E-03


  INCREMENT   124 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.741E-05

  NUMBER OF EQUATIONS =      3006     NUMBER OF RHS =    1
  NUMBER OF FLOATING PT. OPERATIONS =  8.47E+05

     CHECK POINT   START OF SOLVER    

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =          0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.055E-03   TIME AVG. FORCE       5.200E-03
 LARGEST RESIDUAL FORCE            -3.749E-14   AT NODE        485   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF DISP.         1.413E-05   AT NODE        101   DOF  2
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO DISP.        1.071E-06   AT NODE        162   DOF  2
   INSTANCE: BEAM                                                                            
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                     3.458E-02   TIME AVG. MOMENT      4.713E-02
 LARGEST RESIDUAL MOMENT           -5.155E-14   AT NODE         97   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST INCREMENT OF ROTATION      3.228E-07   AT NODE        233   DOF  6
   INSTANCE: BEAM                                                                            
 LARGEST CORRECTION TO ROTATION     5.437E-08   AT NODE        151   DOF  6
   INSTANCE: BEAM                                                                            
          THE MOMENT    EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

  MAXIMUM HALF-INCREMENT FORCE RESIDUAL IS   3.069E-04 AND OCCURS AT NODE        501 INSTANCE BEAM                                                                            
  MAXIMUM HALF-INCREMENT MOMENT RESIDUAL IS   4.406E-04 AND OCCURS AT NODE        500 INSTANCE BEAM                                                                            

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.741E-05,  FRACTION OF STEP COMPLETED   1.00    
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
       USER TIME (SEC)      =   3.9000    
       SYSTEM TIME (SEC)    =  0.30000    
       TOTAL CPU TIME (SEC) =   4.2000    
       WALLCLOCK TIME (SEC) =          7
