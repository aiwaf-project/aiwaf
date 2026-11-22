package com.aiwaf.servlet;

import jakarta.servlet.Filter;
import jakarta.servlet.MultipartConfigElement;
import org.apache.catalina.Context;
import org.apache.catalina.Wrapper;
import org.apache.catalina.startup.Tomcat;
import org.apache.tomcat.util.descriptor.web.FilterDef;
import org.apache.tomcat.util.descriptor.web.FilterMap;

import java.nio.file.Path;

/** The multipart checks in an embedded Tomcat 11. */
class TomcatMultipartTest extends MultipartContainerChecks {
    private Tomcat tomcat;

    @Override
    int start(Path base, Filter filter) throws Exception {
        tomcat = new Tomcat();
        tomcat.setBaseDir(base.toString());
        tomcat.setPort(0);
        Context context = tomcat.addContext("", base.toString());
        Wrapper upload = Tomcat.addServlet(context, "upload", new Upload());
        upload.setMultipartConfigElement(new MultipartConfigElement(base.toString()));
        context.addServletMappingDecoded("/upload", "upload");
        Tomcat.addServlet(context, "raw", new Raw());
        context.addServletMappingDecoded("/raw", "raw");
        FilterDef definition = new FilterDef();
        definition.setFilterName("aiwaf");
        definition.setFilter(filter);
        context.addFilterDef(definition);
        FilterMap map = new FilterMap();
        map.setFilterName("aiwaf");
        map.addURLPattern("/*");
        context.addFilterMap(map);
        tomcat.getConnector();
        tomcat.start();
        return tomcat.getConnector().getLocalPort();
    }

    @Override
    void stop() throws Exception {
        tomcat.stop();
        tomcat.destroy();
    }
}
