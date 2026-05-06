def build_provider(model_cfg):
    provider = model_cfg.provider.lower()
    if provider == 'openai':
        from iga_suite.providers.openai_provider import OpenAIChatProvider
        return OpenAIChatProvider(
            api_key_env=model_cfg.api_key_env,
            model_name=model_cfg.model_name,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
            base_url_env=getattr(model_cfg, 'base_url_env', None),
        )
    if provider == 'anthropic':
        from iga_suite.providers.anthropic_provider import AnthropicMessagesProvider
        return AnthropicMessagesProvider(
            api_key_env=model_cfg.api_key_env,
            model_name=model_cfg.model_name,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
        )
    if provider == 'openai_compatible':
        from iga_suite.providers.openai_provider import OpenAIChatProvider
        return OpenAIChatProvider(
            api_key_env=model_cfg.api_key_env,
            model_name=model_cfg.model_name,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
            base_url_env=getattr(model_cfg, 'base_url_env', None),
        )
    if provider == 'mock':
        from iga_suite.providers.mock_provider import MockProntoQAReasoner
        return MockProntoQAReasoner()
    raise ValueError(f'Unsupported provider: {provider}')
