package com.aiwaf.core;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Declares a handler's query and body inputs, for when {@code init} can't find them all by reading
 * its code, as {@code @AiwafInputs({"q", "page:int", "owner:uuid"})}. Each entry is a name, or a
 * name and a type (int, float, bool, uuid or text; text when left out). With {@code complete} true
 * (the default) the route's list is complete: these, plus anything init sees the code read. On a
 * class, it applies to every handler in it that doesn't have its own.
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.TYPE})
public @interface AiwafInputs {
    String[] value() default {};

    boolean complete() default true;
}
