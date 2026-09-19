package com.aiwaf.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class IsolationForestCoreTest {

    @Test
    void deterministic_scores_with_fixed_seed() {
        double[][] rows = new double[][]{
                {0.0, 0.0},
                {0.1, 0.0},
                {0.2, 0.1},
                {0.3, 0.1},
                {0.4, 0.2},
                {10.0, 10.0}
        };

        IsolationForestCore a = new IsolationForestCore()
                .nEstimators(64)
                .maxSamples(6)
                .contamination(0.2)
                .randomState(42L)
                .fit(rows);
        IsolationForestCore b = new IsolationForestCore()
                .nEstimators(64)
                .maxSamples(6)
                .contamination(0.2)
                .randomState(42L)
                .fit(rows);

        double[] s1 = a.scoreSamples(rows);
        double[] s2 = b.scoreSamples(rows);
        for (int i = 0; i < s1.length; i++) {
            assertEquals(s1[i], s2[i], 1e-12);
        }
    }

    @Test
    void outlier_is_predicted_as_negative_one() {
        double[][] rows = new double[][]{
                {0.0, 0.0},
                {0.1, 0.0},
                {0.2, 0.1},
                {0.3, 0.1},
                {0.4, 0.2},
                {0.5, 0.2},
                {12.0, 12.0}
        };

        IsolationForestCore model = new IsolationForestCore()
                .nEstimators(128)
                .maxSamplesAuto()
                .contamination(0.15)
                .maxFeatures(1.0)
                .randomState(42L)
                .fit(rows);

        int[] pred = model.predict(rows);
        assertEquals(-1, pred[pred.length - 1]);
    }

    @Test
    void contamination_auto_uses_negative_half_offset() {
        double[][] rows = new double[][]{
                {0.0, 0.0},
                {0.1, 0.0},
                {0.2, 0.1},
                {0.3, 0.1}
        };

        IsolationForestCore model = new IsolationForestCore()
                .contaminationAuto()
                .randomState(7L)
                .fit(rows);

        assertEquals(-0.5, model.offset_(), 1e-12);
    }

    @Test
    void warm_start_adds_estimators() {
        double[][] rows = new double[][]{
                {0.0, 0.0},
                {0.1, 0.0},
                {0.2, 0.1},
                {0.3, 0.1},
                {0.4, 0.2},
                {0.5, 0.2}
        };

        IsolationForestCore model = new IsolationForestCore()
                .warmStart(true)
                .randomState(42L)
                .nEstimators(10)
                .fit(rows);
        assertEquals(10, model.estimatorCount());

        model.nEstimators(25).fit(rows);
        assertEquals(25, model.estimatorCount());
    }

    @Test
    void warm_start_rejects_shrinking_estimator_count() {
        double[][] rows = new double[][]{
                {0.0, 0.0},
                {0.1, 0.0},
                {0.2, 0.1},
                {0.3, 0.1},
                {0.4, 0.2}
        };
        IsolationForestCore model = new IsolationForestCore()
                .warmStart(true)
                .nEstimators(10)
                .randomState(11L)
                .fit(rows);
        assertEquals(10, model.estimatorCount());
        model.nEstimators(5);
        assertThrows(IllegalArgumentException.class, () -> model.fit(rows));
    }

    @Test
    void max_samples_auto_and_float_bounds_match_expected_behavior() {
        double[][] rows = new double[500][2];
        for (int i = 0; i < rows.length; i++) {
            rows[i][0] = i;
            rows[i][1] = i % 7;
        }

        IsolationForestCore auto = new IsolationForestCore()
                .maxSamplesAuto()
                .randomState(1L)
                .fit(rows);
        assertEquals(256, auto.maxSamples_());

        IsolationForestCore frac = new IsolationForestCore()
                .maxSamples(0.1)
                .randomState(1L)
                .fit(rows);
        assertEquals(50, frac.maxSamples_());
    }

    @Test
    void decision_function_is_score_minus_offset() {
        double[][] rows = new double[][]{
                {0.0, 0.0},
                {0.1, 0.0},
                {0.2, 0.1},
                {8.0, 8.0}
        };

        IsolationForestCore model = new IsolationForestCore()
                .contamination(0.25)
                .randomState(42L)
                .fit(rows);

        double[] scores = model.scoreSamples(rows);
        double[] decision = model.decisionFunction(rows);
        for (int i = 0; i < rows.length; i++) {
            assertEquals(scores[i] - model.offset_(), decision[i], 1e-12);
        }
    }
}
