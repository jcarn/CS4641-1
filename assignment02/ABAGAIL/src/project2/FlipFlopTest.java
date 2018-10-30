package project2;

import java.text.DecimalFormat;
import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.Instance;

/**
 * A test using the Flip Flop evaluation function
 * @author James Liu
 * @version 1.0
 */
public class FlipFlopTest {
    private static DecimalFormat def = new DecimalFormat("0.000");

    public static void main(String[] args) {
        int N = 80;

        int iterations = 20000;
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FlipFlopEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

       long starttime = System.currentTimeMillis();
       System.out.println("Randomized Hill Climbing");
       RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
       FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
       for (int i = 500; i <= iterations; i += 500) {
           System.out.println("----------" + i + " iterations-----------");
           fit.train();
           System.out.println("RHC: " + ef.value(rhc.getOptimal()));
           System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
       }
       System.out.println("============================");

       System.out.println("Simulated Annealing");
       SimulatedAnnealing sa = new SimulatedAnnealing(100, .8, hcp);
       fit = new FixedIterationTrainer(sa, 500);
       starttime = System.currentTimeMillis();
       for (int i = 500; i <= iterations; i += 500) {
           System.out.println("----------" + i + " iterations-----------");
           fit.train();
           System.out.println("SA: " + ef.value(sa.getOptimal()));
           System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
       }
       System.out.println("============================");



       System.out.println("Genetic Algorithm");
       starttime = System.currentTimeMillis();
       StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200,
               100, 20, gap);
       fit = new FixedIterationTrainer(ga, 500);
       for(int i = 500; i <= iterations; i += 500) {
           System.out.println("----------" + i + " iterations-----------");
           fit.train();
           System.out.println("GA: " + ef.value(ga.getOptimal()));
           System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
       }
       System.out.println("============================");

        System.out.println("MIMIC");
        MIMIC mimic = new MIMIC(200, 5, pop);
        fit = new FixedIterationTrainer(mimic, 500);
        starttime = System.currentTimeMillis();
        for(int i = 500; i <= iterations; i += 500)
        {
            System.out.println("----------" + i + " iterations-----------");
            fit.train();
            System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
            System.out.println("Time : " + def.format((System.currentTimeMillis() - starttime) / Math.pow(10, 3)) + " seconds");
        }
    }
}
