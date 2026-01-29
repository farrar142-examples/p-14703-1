package com.Back.config;

import jakarta.annotation.PostConstruct;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

import java.io.IOException;
import java.net.Proxy;
import java.net.ProxySelector;
import java.net.SocketAddress;
import java.net.URI;
import java.util.Collections;
import java.util.List;

/**
 * REST 클라이언트 설정
 * 
 * Ollama ChatModel은 spring-ai-starter-model-ollama에서 자동 설정됩니다.
 * application.yaml의 spring.ai.ollama 설정을 참조하세요.
 */
@Configuration
public class RestClientConfig {

    @PostConstruct
    public void disableProxy() {
        // 시스템 프록시 설정 비활성화 (Ollama 로컬 연결 시에도 유용)
        System.setProperty("java.net.useSystemProxies", "false");
        System.setProperty("http.proxyHost", "");
        System.setProperty("https.proxyHost", "");
        
        // 커스텀 ProxySelector 등록 - 항상 DIRECT 연결 사용
        ProxySelector.setDefault(new ProxySelector() {
            @Override
            public List<Proxy> select(URI uri) {
                return Collections.singletonList(Proxy.NO_PROXY);
            }

            @Override
            public void connectFailed(URI uri, SocketAddress sa, IOException ioe) {
                // 실패 시 아무것도 하지 않음
            }
        });
    }

    @Bean
    public RestTemplate restTemplate() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setProxy(Proxy.NO_PROXY);
        factory.setConnectTimeout(30000);
        factory.setReadTimeout(60000);
        return new RestTemplate(factory);
    }
}
