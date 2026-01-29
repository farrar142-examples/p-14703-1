package com.Back;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class BackApplicationTests {
	@Autowired
	ChatModel chatModel;

	@Autowired
	EmbeddingModel embeddingModel;

	@Test
	@DisplayName("EmbeddingModel 테스트")
	void t1() {
		float[] embedding = embeddingModel.embed("Hello, world!");
		assertNotNull(embedding);
		System.out.println(embedding.length);
	}

	@Test
	@DisplayName("ChatModel 테스트")
	void t2(){
		String response = chatModel.call(
				UserMessage.builder()
						.text("Hello, how are you?")
				.build()
		);
		assertNotNull(response);
		System.out.println(response);
	}

}
