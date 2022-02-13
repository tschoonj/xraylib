package com.github.tschoonj.xraylib;

import java.io.DataOutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.prefs.Preferences;

public class GoogleAnalyticsThread extends Thread {

    private static final String GOOGLE_ANALYTICS_ENDPOINT = "https://www.google-analytics.com/collect";
    private static final String GOOGLE_ANALYTICS_TRACKING_ID = "UA-42595764-5";
    private static final String GOOGLE_ANALYTICS_APPLICATION_NAME = "xraylib";
    private static final String GOOGLE_ANALYTICS_APPLICATION_VERSION = "4.1.2";
    private static final String GOOGLE_ANALYTICS_HIT_TYPE = "event";

    public void run() {
      Map<String, String> payload = new HashMap<String, String>();
      payload.put("v", "1");
      payload.put("tid", GOOGLE_ANALYTICS_TRACKING_ID);
      payload.put("t", GOOGLE_ANALYTICS_HIT_TYPE);
      payload.put("an", GOOGLE_ANALYTICS_APPLICATION_NAME);
      payload.put("av", GOOGLE_ANALYTICS_APPLICATION_VERSION);

      if (System.getenv().containsKey("CI")) {
        payload.put("cid", "60220817-0a15-49ce-b581-9cab2b225e7d");
        payload.put("ec", "CI-java");
      } else {
        Preferences prefs = Preferences.userNodeForPackage(GoogleAnalyticsThread.class);
        payload.put("cid", prefs.get("uuid", UUID.randomUUID().toString()));
        payload.put("ec", "java");
      }

      payload.put("ea", "import");
      payload.put("el", String.format("xraylib-%s-Java-%s-%s", GOOGLE_ANALYTICS_APPLICATION_VERSION, System.getProperty("java.runtime.name"), System.getProperty("java.runtime.version")));
      payload.put("ua", "Opera/9.80 (Windows NT 6.0) Presto/2.12.388 Version/12.14");

      String url = GOOGLE_ANALYTICS_ENDPOINT;
      try {
        StringBuilder postData = new StringBuilder();
        for (Map.Entry<String, String> parameter : payload.entrySet()) {
          if (postData.length() != 0)
            postData.append('&');
          final String encodedKey = URLEncoder.encode(parameter.getKey().toString(), "UTF-8");
          final String encodedValue = URLEncoder.encode(parameter.getValue().toString(), "UTF-8");
          postData.append(encodedKey);
          postData.append("=");
          postData.append(encodedValue);
        }
        byte[] postDataBytes = postData.toString().getBytes("UTF-8");
  
        HttpURLConnection connection = (HttpURLConnection) new URL(url).openConnection();
        connection.setRequestMethod("POST");
        connection.setConnectTimeout(1000);
        connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
        connection.setDoOutput(true);
        connection.setUseCaches(false);
        connection.setRequestProperty("Content-Length", String.valueOf(postDataBytes.length));

        try(DataOutputStream wr = new DataOutputStream(connection.getOutputStream())) {
          wr.write(postDataBytes);
        }

        connection.getResponseCode();
      } catch (Exception e) {
        System.err.println("GoogleAnalyticsThread exception caught: " + e.getMessage());
      }
    }
  public static void main(String[] args) {
    Thread thread = new GoogleAnalyticsThread();
    thread.start();
    try {
        thread.join();
    } catch (InterruptedException e) {
        System.err.println("Thread got interrupted");
    }
  }

}
