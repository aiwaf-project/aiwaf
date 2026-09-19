package com.aiwaf.core;

import java.util.concurrent.atomic.AtomicReference;

public final class LazyModelProviderCore {
    private final String modelPath;
    private final AtomicReference<TrainedModelCore> cached = new AtomicReference<>();
    private volatile boolean attempted;

    public LazyModelProviderCore(String modelPath) {
        this.modelPath = modelPath;
    }

    public TrainedModelCore get() {
        TrainedModelCore current = cached.get();
        if (current != null) {
            return current;
        }
        if (attempted) {
            return null;
        }
        synchronized (this) {
            current = cached.get();
            if (current != null) {
                return current;
            }
            if (attempted) {
                return null;
            }
            attempted = true;
            TrainedModelCore loaded = ModelArtifactIoCore.load(modelPath);
            if (loaded != null) {
                cached.set(loaded);
            }
            return loaded;
        }
    }

    public void set(TrainedModelCore model) {
        if (model != null) {
            cached.set(model);
            attempted = true;
        }
    }
}
