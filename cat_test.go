package opencat_api

import (
	"context"
	"io"
	"os"
	"testing"
)

func client() *Client {
	if os.Getenv("TOKEN") == "" {
		panic("TOKEN not set")
	}
	return NewClient(os.Getenv("TOKEN"))
}

func TestChat(t *testing.T) {
	c := client()
	resp, err := c.Chat(
		context.Background(),
		ChatRequest{
			Model:       ChatModelGPT3Dot5Turbo,
			Temperature: 1,
			MaxTokens:   4096,
			Stream:      false,
			Messages: []Message{
				{
					Role:    "system",
					Content: "You are a helpful assistant.",
				},
				{
					Role:    "user",
					Content: "Hello!",
				},
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("resp: %s", resp.Choices[0].Message.Content)
}

func TestChatClaude(t *testing.T) {
	c := client()
	resp, err := c.Chat(
		context.Background(),
		ChatRequest{
			Model:       ChatModelClaudeInstant1,
			Temperature: 1,
			MaxTokens:   4096,
			Stream:      false,
			Messages: []Message{
				{
					Role:    "system",
					Content: "You are a helpful assistant.",
				},
				{
					Role:    "user",
					Content: "Hello!",
				},
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("resp: %s", resp.Choices[0].Message.Content)
}

func TestStreamChat(t *testing.T) {
	c := client()
	content := ""
	err := c.StreamChat(
		context.Background(),
		ChatRequest{
			Model:       ChatModelGPT3Dot5Turbo,
			Temperature: 1,
			MaxTokens:   4096,
			Stream:      true,
			Messages: []Message{
				{
					Role:    "system",
					Content: "You are a helpful assistant.",
				},
				{
					Role:    "user",
					Content: "Hello!",
				},
			},
		},
		func(delta string, done bool) {
			content += delta
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("resp: %s", content)
}

func TestStreamChatClaude(t *testing.T) {
	c := client()
	content := ""
	err := c.StreamChat(
		context.Background(),
		ChatRequest{
			Model:       ChatModelClaudeInstant1,
			Temperature: 1,
			MaxTokens:   4096,
			Stream:      true,
			Messages: []Message{
				{
					Role:    "system",
					Content: "You are a helpful assistant.",
				},
				{
					Role:    "user",
					Content: "Hello!",
				},
			},
		},
		func(delta string, done bool) {
			content += delta
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("resp: %s", content)
}

func TestChatImage(t *testing.T) {
	c := client()
	img, err := os.Open("testdata/1.jpeg")
	if err != nil {
		t.Fatal(err)
	}

	content := ""
	err = c.StreamChat(
		context.Background(),
		ChatRequest{
			Model:       ChatModelGPT4VisionPreview,
			Temperature: 1,
			MaxTokens:   4096,
			Stream:      true,
			Messages: []Message{
				{
					Role:    "system",
					Content: "You are a helpful assistant.",
				},
				{
					Role:    "user",
					Content: "Describe this image",
					Images:  []Image{NewImage(img)},
				},
			},
		},
		func(delta string, done bool) {
			content += delta
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("resp: %s", content)
}

func TestGenImage(t *testing.T) {
	c := client()
	imgs, err := c.Image(
		context.Background(), ImageRequest{
			Width:  256,
			Height: 256,
			Num:    1,
			Model:  ImageModelDallE2,
			Prompt: "a kitten in a basket",
			DallE: DallEParams{
				Quality: "standard",
				Style:   "vivid",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	err = os.WriteFile("output/gen.jpg", imgs[0], 0644)
	if err != nil {
		t.Fatal(err)
	}
}

func writeFile(name string, in io.Reader) error {
	f, err := os.Create(name)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, in)
	return err
}

func TestGenSpeech(t *testing.T) {
	c := client()
	speech, err := c.Speech(
		context.Background(),
		SpeechRequest{
			Input: "Hello! How are you!",
			Voice: "alloy",
			Model: SpeechModelTTS1,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer speech.Close()

	err = writeFile("output/speech.wav", speech)
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenSpeechAzure(t *testing.T) {
	c := client()
	speech, err := c.Speech(
		context.Background(),
		SpeechRequest{
			Input: "你好啊，李银河！",
			Voice: "zh-CN-XiaoxiaoNeural",
			Model: SpeechModelAzure,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer speech.Close()

	err = writeFile("output/speech2.wav", speech)
	if err != nil {
		t.Fatal(err)
	}
}
