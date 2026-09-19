package com.aiwaf.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.io.Serializable;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public final class IsolationForestCore {
    private static final double EULER_GAMMA = 0.5772156649015329;

    private int nEstimators = 100;
    private String maxSamples = "auto";
    private String contamination = "auto";
    private String maxFeatures = "1.0";
    private boolean bootstrap = false;
    private int nJobs = 1;
    private Long randomState = null;
    private int verbose = 0;
    private boolean warmStart = false;

    private final List<IsolationTree> estimators = new ArrayList<>();
    private int maxSamplesActual = -1;
    private int nFeaturesIn = -1;
    private double offset = -0.5;
    private boolean fitted = false;

    public IsolationForestCore() {}

    public IsolationForestCore nEstimators(int value) {
        if (value < 1) throw new IllegalArgumentException("n_estimators must be >= 1");
        this.nEstimators = value;
        return this;
    }

    public IsolationForestCore maxSamplesAuto() {
        this.maxSamples = "auto";
        return this;
    }

    public IsolationForestCore maxSamples(int value) {
        if (value < 1) throw new IllegalArgumentException("max_samples int must be >= 1");
        this.maxSamples = String.valueOf(value);
        return this;
    }

    public IsolationForestCore maxSamples(double value) {
        if (!(value > 0.0 && value <= 1.0)) {
            throw new IllegalArgumentException("max_samples float must be in (0, 1]");
        }
        this.maxSamples = String.valueOf(value);
        return this;
    }

    public IsolationForestCore contaminationAuto() {
        this.contamination = "auto";
        return this;
    }

    public IsolationForestCore contamination(double value) {
        if (!(value > 0.0 && value <= 0.5)) {
            throw new IllegalArgumentException("contamination must be in (0, 0.5]");
        }
        this.contamination = String.valueOf(value);
        return this;
    }

    public IsolationForestCore maxFeatures(int value) {
        if (value < 1) throw new IllegalArgumentException("max_features int must be >= 1");
        this.maxFeatures = String.valueOf(value);
        return this;
    }

    public IsolationForestCore maxFeatures(double value) {
        if (!(value > 0.0 && value <= 1.0)) {
            throw new IllegalArgumentException("max_features float must be in (0, 1]");
        }
        this.maxFeatures = String.valueOf(value);
        return this;
    }

    public IsolationForestCore bootstrap(boolean value) {
        this.bootstrap = value;
        return this;
    }

    public IsolationForestCore nJobs(int value) {
        if (value == 0) throw new IllegalArgumentException("n_jobs cannot be 0");
        this.nJobs = value;
        return this;
    }

    public IsolationForestCore randomState(long value) {
        this.randomState = value;
        return this;
    }

    public IsolationForestCore verbose(int value) {
        this.verbose = value;
        return this;
    }

    public IsolationForestCore warmStart(boolean value) {
        this.warmStart = value;
        return this;
    }

    public IsolationForestCore fit(double[][] x) {
        validateX(x);
        nFeaturesIn = x[0].length;
        maxSamplesActual = resolveMaxSamples(x.length);
        int maxDepth = (int) Math.ceil(log2(Math.max(maxSamplesActual, 2)));

        int targetTrees = nEstimators;
        if (!warmStart) {
            estimators.clear();
        } else if (estimators.size() > targetTrees) {
            throw new IllegalArgumentException(
                    "n_estimators=" + targetTrees + " must be >= existing estimator count="
                            + estimators.size() + " when warmStart=true"
            );
        }

        int toBuild = targetTrees - estimators.size();
        if (toBuild > 0) {
            long baseSeed = randomState == null ? System.nanoTime() : randomState;
            int workers = resolveWorkers(nJobs);
            if (workers == 1 || toBuild == 1) {
                for (int i = 0; i < toBuild; i++) {
                    Random rnd = new Random(baseSeed + estimators.size() + i);
                    estimators.add(buildSingleEstimator(x, maxDepth, rnd));
                }
            } else {
                ExecutorService pool = Executors.newFixedThreadPool(workers);
                try {
                    List<Future<IsolationTree>> futures = new ArrayList<>();
                    for (int i = 0; i < toBuild; i++) {
                        final int idx = i;
                        futures.add(pool.submit(new Callable<>() {
                            @Override
                            public IsolationTree call() {
                                Random rnd = new Random(baseSeed + estimators.size() + idx);
                                return buildSingleEstimator(x, maxDepth, rnd);
                            }
                        }));
                    }
                    for (Future<IsolationTree> future : futures) {
                        estimators.add(future.get());
                    }
                } catch (Exception ex) {
                    throw new IllegalStateException("failed building isolation forest", ex);
                } finally {
                    pool.shutdown();
                }
            }
        }

        fitted = true;
        if ("auto".equals(contamination)) {
            offset = -0.5;
        } else {
            double c = Double.parseDouble(contamination);
            double[] scores = scoreSamples(x);
            offset = percentile(scores, 100.0 * c);
        }
        return this;
    }

    public int[] predict(double[][] x) {
        ensureFitted();
        double[] decision = decisionFunction(x);
        int[] out = new int[decision.length];
        Arrays.fill(out, 1);
        for (int i = 0; i < decision.length; i++) {
            if (decision[i] < 0.0) out[i] = -1;
        }
        return out;
    }

    public double[] decisionFunction(double[][] x) {
        ensureFitted();
        double[] scores = scoreSamples(x);
        for (int i = 0; i < scores.length; i++) {
            scores[i] -= offset;
        }
        return scores;
    }

    // sklearn-like: lower means more abnormal
    public double[] scoreSamples(double[][] x) {
        ensureFitted();
        validatePredictX(x);
        double[] anomalyScores = computeAnomalyScores(x);
        for (int i = 0; i < anomalyScores.length; i++) {
            anomalyScores[i] = -anomalyScores[i];
        }
        return anomalyScores;
    }

    // Compatibility helpers for prior static API
    public static Model fit(double[][] rows, int nTrees, int sampleSize, long randomSeed) {
        IsolationForestCore estimator = new IsolationForestCore()
                .nEstimators(nTrees)
                .maxSamples(sampleSize)
                .contamination(0.05)
                .randomState(randomSeed)
                .fit(rows);
        return new Model(new ArrayList<>(estimator.estimators), estimator.maxSamplesActual);
    }

    public static double score(Model model, double[] row) {
        if (model == null) throw new IllegalArgumentException("model must not be null");
        if (row == null || row.length == 0) throw new IllegalArgumentException("row must not be empty");
        double score = 0.0;
        for (IsolationTree tree : model.trees()) {
            score += anomalyScoreForTree(tree, row, model.sampleSize());
        }
        return score / model.trees().size();
    }

    public static double[] scoreAll(Model model, double[][] rows) {
        double[] out = new double[rows.length];
        for (int i = 0; i < rows.length; i++) {
            out[i] = score(model, rows[i]);
        }
        return out;
    }

    public static double thresholdFromContamination(double[] scores, double contamination) {
        if (scores == null || scores.length == 0) return 1.0;
        double clamped = Math.max(0.0, Math.min(0.5, contamination));
        if (clamped <= 0.0) return Double.POSITIVE_INFINITY;
        double[] copy = Arrays.copyOf(scores, scores.length);
        Arrays.sort(copy);
        int idx = (int) Math.floor((1.0 - clamped) * (copy.length - 1));
        idx = Math.max(0, Math.min(copy.length - 1, idx));
        return copy[idx];
    }

    public static int countAnomalies(double[] scores, double threshold) {
        int count = 0;
        for (double score : scores) {
            if (score >= threshold) count++;
        }
        return count;
    }

    public int nFeaturesIn() {
        ensureFitted();
        return nFeaturesIn;
    }

    public int maxSamples_() {
        ensureFitted();
        return maxSamplesActual;
    }

    public double offset_() {
        ensureFitted();
        return offset;
    }

    public int estimatorCount() {
        ensureFitted();
        return estimators.size();
    }

    private IsolationTree buildSingleEstimator(double[][] x, int maxDepth, Random rnd) {
        int[] featureSubset = sampleFeatureSubset(nFeaturesIn, resolveMaxFeatures(nFeaturesIn), rnd);
        double[][] sample = bootstrap
                ? sampleWithReplacement(x, maxSamplesActual, rnd)
                : sampleWithoutReplacement(x, maxSamplesActual, rnd);
        Node root = buildTree(sample, 0, maxDepth, featureSubset, rnd);
        return new IsolationTree(root, featureSubset);
    }

    private double[] computeAnomalyScores(double[][] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double s = 0.0;
            for (IsolationTree tree : estimators) {
                s += anomalyScoreForTree(tree, x[i], maxSamplesActual);
            }
            out[i] = s / estimators.size();
        }
        return out;
    }

    private static double anomalyScoreForTree(IsolationTree tree, double[] row, int sampleSize) {
        double pathLength = pathLength(row, tree.root(), 0);
        double cn = c(sampleSize);
        if (cn <= 0.0) return 0.0;
        return Math.pow(2.0, -pathLength / cn);
    }

    private static Node buildTree(double[][] data, int depth, int maxDepth, int[] featureSubset, Random random) {
        if (data.length <= 1 || depth >= maxDepth) {
            return Node.leaf(data.length);
        }

        int[] shuffled = Arrays.copyOf(featureSubset, featureSubset.length);
        shuffle(shuffled, random);
        int chosenFeature = -1;
        double min = 0.0;
        double max = 0.0;
        for (int feature : shuffled) {
            double[] minMax = minMax(data, feature);
            if (minMax[0] < minMax[1]) {
                chosenFeature = feature;
                min = minMax[0];
                max = minMax[1];
                break;
            }
        }
        if (chosenFeature < 0) {
            return Node.leaf(data.length);
        }

        double split = min + random.nextDouble() * (max - min);
        List<double[]> left = new ArrayList<>();
        List<double[]> right = new ArrayList<>();
        for (double[] row : data) {
            double v = row[chosenFeature];
            if (!Double.isFinite(v) || v >= split) {
                right.add(row);
            } else {
                left.add(row);
            }
        }
        if (left.isEmpty() || right.isEmpty()) {
            return Node.leaf(data.length);
        }

        Node leftNode = buildTree(left.toArray(new double[0][]), depth + 1, maxDepth, featureSubset, random);
        Node rightNode = buildTree(right.toArray(new double[0][]), depth + 1, maxDepth, featureSubset, random);
        return Node.branch(chosenFeature, split, leftNode, rightNode);
    }

    private static double[] minMax(double[][] data, int feature) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double[] row : data) {
            double v = row[feature];
            if (!Double.isFinite(v)) continue;
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        if (min == Double.POSITIVE_INFINITY) {
            return new double[]{0.0, 0.0};
        }
        return new double[]{min, max};
    }

    private static double pathLength(double[] row, Node node, int depth) {
        if (node.leaf()) {
            return depth + c(node.leafSize());
        }
        double v = row[node.feature()];
        if (Double.isFinite(v) && v < node.split()) {
            return pathLength(row, node.left(), depth + 1);
        }
        return pathLength(row, node.right(), depth + 1);
    }

    private static double c(int n) {
        if (n <= 1) return 0.0;
        if (n == 2) return 1.0;
        return 2.0 * (Math.log(n - 1) + EULER_GAMMA) - (2.0 * (n - 1) / n);
    }

    private static double[][] sampleWithoutReplacement(double[][] rows, int size, Random random) {
        int[] indices = new int[rows.length];
        for (int i = 0; i < rows.length; i++) indices[i] = i;
        shuffle(indices, random);
        double[][] sample = new double[size][];
        for (int i = 0; i < size; i++) sample[i] = rows[indices[i]];
        return sample;
    }

    private static double[][] sampleWithReplacement(double[][] rows, int size, Random random) {
        double[][] sample = new double[size][];
        for (int i = 0; i < size; i++) {
            sample[i] = rows[random.nextInt(rows.length)];
        }
        return sample;
    }

    private static int[] sampleFeatureSubset(int totalFeatures, int subsetSize, Random random) {
        int[] all = new int[totalFeatures];
        for (int i = 0; i < totalFeatures; i++) all[i] = i;
        shuffle(all, random);
        return Arrays.copyOf(all, Math.max(1, Math.min(subsetSize, totalFeatures)));
    }

    private int resolveMaxSamples(int nSamples) {
        if ("auto".equals(maxSamples)) {
            return Math.min(256, nSamples);
        }
        if (maxSamples.contains(".")) {
            double f = Double.parseDouble(maxSamples);
            return Math.max(1, Math.min(nSamples, (int) Math.floor(f * nSamples)));
        }
        int v = Integer.parseInt(maxSamples);
        return Math.min(v, nSamples);
    }

    private int resolveMaxFeatures(int nFeatures) {
        if (maxFeatures.contains(".")) {
            double f = Double.parseDouble(maxFeatures);
            return Math.max(1, Math.min(nFeatures, (int) Math.floor(f * nFeatures)));
        }
        int v = Integer.parseInt(maxFeatures);
        return Math.max(1, Math.min(v, nFeatures));
    }

    private static int resolveWorkers(int nJobs) {
        if (nJobs == -1) {
            return Math.max(1, Runtime.getRuntime().availableProcessors());
        }
        return Math.max(1, nJobs);
    }

    private static void shuffle(int[] arr, Random random) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    private static double log2(double v) {
        return Math.log(v) / Math.log(2.0);
    }

    private static double percentile(double[] values, double p) {
        if (values.length == 0) return Double.NaN;
        double[] sorted = Arrays.copyOf(values, values.length);
        Arrays.sort(sorted);
        double rank = (p / 100.0) * (sorted.length - 1);
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);
        if (lo == hi) return sorted[lo];
        double w = rank - lo;
        return sorted[lo] * (1.0 - w) + sorted[hi] * w;
    }

    private void validateX(double[][] x) {
        if (x == null || x.length == 0) throw new IllegalArgumentException("X must not be empty");
        if (x[0] == null || x[0].length == 0) throw new IllegalArgumentException("X must have at least one feature");
        int features = x[0].length;
        for (double[] row : x) {
            if (row == null || row.length != features) throw new IllegalArgumentException("all rows must have same feature count");
        }
    }

    private void validatePredictX(double[][] x) {
        if (x == null || x.length == 0) throw new IllegalArgumentException("X must not be empty");
        for (double[] row : x) {
            if (row == null || row.length != nFeaturesIn) {
                throw new IllegalArgumentException("row feature count mismatch");
            }
        }
    }

    private void ensureFitted() {
        if (!fitted || estimators.isEmpty()) {
            throw new IllegalStateException("IsolationForestCore is not fitted");
        }
    }

    public record Model(List<IsolationTree> trees, int sampleSize) implements Serializable {}

    public record IsolationTree(Node root, int[] featureSubset) implements Serializable {}

    public record Node(
            boolean leaf,
            int leafSize,
            int feature,
            double split,
            Node left,
            Node right
    ) implements Serializable {
        static Node leaf(int leafSize) {
            return new Node(true, leafSize, -1, 0.0, null, null);
        }

        static Node branch(int feature, double split, Node left, Node right) {
            return new Node(false, 0, feature, split, left, right);
        }
    }
}
