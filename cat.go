package opencat_api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"strings"
)

const baseURL = "https://api.opencat.app"

type ImageModel string

var (
	ImageModelDallE2            ImageModel = "dall-e-2"
	ImageModelDallE3            ImageModel = "dall-e-3"
	ImageModelStableDiffusionXL ImageModel = "stable_diffusion_xl"
)

type ChatModel string

var (
	// OpenAI
	ChatModelGPT3Dot5Turbo     ChatModel = "gpt-3.5-turbo"
	ChatModelGPT3Dot5Turbo16K  ChatModel = "gpt-3.5-turbo-16k"
	ChatModelGPT4              ChatModel = "gpt-4"
	ChatModelGPT432K           ChatModel = "gpt-4-32k"
	ChatModelGPT4Turbo         ChatModel = "gpt-4-1106-preview"
	ChatModelGPT4VisionPreview ChatModel = "gpt-4-vision-preview"
	// Anthropic Claude
	ChatModelClaudeInstant1 ChatModel = "claude-instant-v1"
	ChatModelClaude2        ChatModel = "claude-2.1"
	// Google GEMINI
	ChatModelGEMINIPro       ChatModel = "gemini-pro"
	ChatModelGEMINIProVision ChatModel = "gemini-pro-vision"
	// 百度文心一言
	ChatModelERNIEBot      ChatModel = "ERNIE-Bot"
	ChatModelERNIEBotTurbo ChatModel = "ERNIE-Bot-Turbo"
	ChatModelERNIEBot4     ChatModel = "ERNIE-Bot-4"
	// 阿里通义千问
	ChatModelQWENTurbo ChatModel = "qwen-turbo"
	ChatModelQWENPlus  ChatModel = "qwen-plus"
	// 讯飞星火大模型
	ChatModelSparkDeskV1 ChatModel = "SparkDesk-V1.5"
	ChatModelSparkDeskV2 ChatModel = "SparkDesk-V2.0"
	ChatModelSparkDeskV3 ChatModel = "SparkDesk-V3.0"
)

type SpeechModel string

var (
	SpeechModelTTS1  SpeechModel = "tts-1"
	SpeechModelAzure SpeechModel = "__azure"
)

type Role string

var (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type Message struct {
	Role    Role    `json:"role"`
	Content string  `json:"content"`
	Images  []Image `json:"images,omitempty"`
}

type ChatRequest struct {
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"maxTokens,omitempty"`
	Model       ChatModel `json:"model"`
	Stream      bool      `json:"stream,omitempty"`
	Messages    []Message `json:"messages"`
}

type ChatResponseChoice struct {
	Index   int `json:"index"`
	Message struct {
		Content string `json:"content"`
		Role    Role   `json:"role"`
	} `json:"message"`
	FinishReason string `json:"finish_reason"`
}

type ChatResponse struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []ChatResponseChoice `json:"choices"`
	Usage   Usage                `json:"usage"`
}

type DallEParams struct {
	Quality string `json:"quality"`
	Style   string `json:"style"`
}

type StableDiffusionXLParams struct {
	Steps       int    `json:"steps"`
	Sampler     string `json:"sampler"`
	StylePreset string `json:"style_preset"`
	Scale       int    `json:"scale"`
}

type ImageRequest struct {
	Width             int                     `json:"width"`
	Height            int                     `json:"height"`
	Num               int                     `json:"num"`
	Model             ImageModel              `json:"model"`
	Prompt            string                  `json:"prompt"`
	NegativePrompt    string                  `json:"negativePrompt"`
	DallE             DallEParams             `json:"dallE,omitempty"`
	StableDiffusionXL StableDiffusionXLParams `json:"stable_diffusion_xl,omitempty"`
}

type SpeechRequest struct {
	Input string      `json:"input"`
	Voice string      `json:"voice"`
	Model SpeechModel `json:"model"`
}

type Usage struct {
	ID      string             `json:"id"`
	Limit   int                `json:"limit"`
	Product string             `json:"product"`
	Usage   map[string]float32 `json:"usage"`
}

type APIError struct {
	HTTPStatusCode int
	Body           string
}

func NewAPIError(resp *http.Response) *APIError {
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return &APIError{resp.StatusCode, string(body)}
}

func (e *APIError) Error() string {
	return fmt.Sprintf("API returned error: code=%d, body=%s", e.HTTPStatusCode, e.Body)
}

type Client struct {
	token  string
	client http.Client
}

func NewClient(token string) *Client {
	return &Client{
		token: token,
	}
}

type Image struct {
	r io.Reader
}

func NewImage(r io.Reader) Image {
	return Image{r: r}
}

func (img *Image) MarshalJSON() ([]byte, error) {
	buf := bytes.NewBuffer(nil)
	buf.WriteString(`"data:image/jpeg;base64,`)
	enc := base64.NewEncoder(base64.StdEncoding, buf)
	_, err := io.Copy(enc, img.r)
	if err != nil {
		return nil, err
	}
	err = enc.Close()
	if err != nil {
		return nil, err
	}
	buf.WriteString(`"`)
	return buf.Bytes(), nil
}

func (c *Client) addHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("User-Agent", "OpenCat/424 CFNetwork/1490.0.4 Darwin/23.2.0")
	req.Header.Set("Accept", "*/*")
}

func (c *Client) chat(ctx context.Context, chat ChatRequest) (*http.Response, error) {
	var req *http.Request
	var err error
	if strings.HasPrefix(string(chat.Model), "claude") {
		req, err = c.claudeRequest(ctx, chat)
		if err != nil {
			return nil, err
		}
	} else {
		var body []byte
		body, err = json.Marshal(chat)
		if err != nil {
			return nil, err
		}
		req, err = http.NewRequestWithContext(ctx, "POST", baseURL+"/1/chat", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		c.addHeaders(req)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// Chat generates a response from a list of messages.
func (c *Client) Chat(ctx context.Context, chat ChatRequest) (_ ChatResponse, err error) {
	if chat.Stream {
		err = errors.New("use StreamChat for streaming chat instead")
		return
	}

	resp, err := c.chat(ctx, chat)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		err = NewAPIError(resp)
		return
	}

	if strings.HasPrefix(string(chat.Model), "claude") {
		var r struct {
			Type       string `json:"type"`
			ID         string `json:"id"`
			Model      string `json:"model"`
			Completion string `json:"completion"`
			StopReason string `json:"stop_reason"`
		}
		err = json.NewDecoder(resp.Body).Decode(&r)
		if err != nil {
			return
		}

		cr := ChatResponse{
			ID:      r.ID,
			Object:  "chat.completion",
			Model:   r.Model,
			Choices: []ChatResponseChoice{{}},
		}
		cr.Choices[0].Message.Role = RoleAssistant
		cr.Choices[0].Message.Content = r.Completion
		cr.Choices[0].FinishReason = r.StopReason
		return cr, nil
	} else {
		var r ChatResponse
		err = json.NewDecoder(resp.Body).Decode(&r)
		if err != nil {
			return
		}
		return r, nil
	}
}

// StreamChat generates a response from a list of messages, and streams the response.
func (c *Client) StreamChat(ctx context.Context, chat ChatRequest, fn func(delta string, done bool)) error {
	if !chat.Stream {
		return errors.New("use Chat for non-streaming chat instead")
	}
	resp, err := c.chat(ctx, chat)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		return NewAPIError(resp)
	}

	r := bufio.NewReader(resp.Body)
	for {
		line, err := r.ReadBytes('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}

		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		line = bytes.TrimPrefix(line, []byte("data: "))
		if bytes.Equal(line, []byte("[DONE]")) {
			break
		}

		var delta struct {
			Type         string `json:"type"`
			Model        string `json:"model"`
			Delta        string `json:"delta"`
			Completion   string `json:"completion"`
			FinishReason string `json:"finishReason"`
		}
		err = json.Unmarshal(line, &delta)
		if err != nil {
			return err
		}

		if delta.Type != "" && delta.Type != "completion" {
			continue
		}

		text := delta.Delta
		if delta.Completion != "" {
			text = delta.Completion
		}
		fn(text, false)
	}
	fn("", true)
	return nil
}

func (c *Client) claudeRequest(ctx context.Context, chat ChatRequest) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v1/complete", nil)
	if err != nil {
		return nil, err
	}
	c.addHeaders(req)

	var prompt strings.Builder
	for _, msg := range chat.Messages {
		switch msg.Role {
		case RoleSystem, RoleUser:
			prompt.WriteString("\n\nHuman: ")
		case RoleAssistant:
			prompt.WriteString("\n\nAssistant: ")
		}
		prompt.WriteString(msg.Content)
	}
	prompt.WriteString("\n\nAssistant:")

	body := map[string]any{
		"model":                chat.Model,
		"temperature":          chat.Temperature,
		"stream":               chat.Stream,
		"max_tokens_to_sample": chat.MaxTokens,
		"prompt":               prompt.String(),
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	return req, nil
}

// Image generates an image from a text prompt.
func (c *Client) Image(ctx context.Context, image ImageRequest) ([][]byte, error) {
	body, err := json.Marshal(image)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/1/images/generations", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	c.addHeaders(req)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, NewAPIError(resp)
	}

	var images struct {
		ImageData [][]byte `json:"image_data"`
	}
	err = json.NewDecoder(resp.Body).Decode(&images)
	if err != nil {
		return nil, err
	}

	return images.ImageData, nil
}

// Speech generates speech from a text input.
func (c *Client) Speech(ctx context.Context, speech SpeechRequest) ([]byte, error) {
	if speech.Model == SpeechModelAzure {
		return c.azureSpeech(ctx, speech)
	}

	body, err := json.Marshal(speech)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v1/audio/speech", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	c.addHeaders(req)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, NewAPIError(resp)
	}

	return io.ReadAll(resp.Body)
}

func (c *Client) azureSpeech(ctx context.Context, speech SpeechRequest) ([]byte, error) {
	body := fmt.Sprintf(
		`
<speak version="1.0" xml:lang="en-US">
<voice xml:lang="en-US" name="%s">%s</voice>
</speak>
`, speech.Voice, html.EscapeString(speech.Input),
	)
	req, _ := http.NewRequestWithContext(ctx, "POST", baseURL+"/cognitiveservices/v1", strings.NewReader(body))
	c.addHeaders(req)
	req.Header.Set("X-Microsoft-OutputFormat", "audio-16khz-128kbitrate-mono-mp3")
	req.Header.Set("X-Region", "eastasia")
	req.Header.Set("Content-Type", "application/ssml+xml")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, NewAPIError(resp)
	}

	return io.ReadAll(resp.Body)
}

// Usage returns the current usage of the API.
func (c *Client) Usage(ctx context.Context) ([]Usage, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", baseURL+"/1.1/me/usage", nil)
	if err != nil {
		return nil, err
	}
	c.addHeaders(req)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, NewAPIError(resp)
	}

	var data struct {
		Data []Usage `json:"data"`
	}
	err = json.NewDecoder(resp.Body).Decode(&data)
	if err != nil {
		return nil, err
	}

	return data.Data, nil
}
