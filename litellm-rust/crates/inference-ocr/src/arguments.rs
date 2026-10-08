use litellm_llms::base_llm::ocr::error::Error;

use super::provider_config::{OcrConfigKind, resolve_provider_config};

const COMMON_OPTION_FIELDS: &[&str] = &["req_format", "max_response_bytes"];
const AZURE_AUTH_OPTION_FIELDS: &[&str] = &[
    "azure_ad_token",
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_scope",
    "azure_authority_host",
    "azure_credential",
    "azure_federated_token_file",
    "enable_azure_ad_token_refresh",
];
const AWS_AUTH_OPTION_FIELDS: &[&str] = &[
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_session_name",
    "aws_profile_name",
    "aws_role_name",
    "aws_web_identity_token",
    "aws_sts_endpoint",
    "aws_external_id",
];
const VERTEX_AUTH_OPTION_FIELDS: &[&str] = &[
    "vertex_credentials",
    "vertex_ai_credentials",
    "vertex_project",
    "vertex_ai_project",
    "vertex_location",
    "vertex_ai_location",
];

pub fn is_supported_request(model: &str, custom_llm_provider: Option<&str>) -> bool {
    resolve_provider_config(model, custom_llm_provider).is_ok()
}

pub fn owned_option_names(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<Vec<&'static str>, Error> {
    let (_, config) = resolve_provider_config(model, custom_llm_provider)?;
    let auth_fields: &[&str] = match config {
        OcrConfigKind::AwsTextract | OcrConfigKind::AwsTextractAnalyze => AWS_AUTH_OPTION_FIELDS,
        OcrConfigKind::AzureAi
        | OcrConfigKind::AzureDocumentIntelligence
        | OcrConfigKind::AzureCohere => AZURE_AUTH_OPTION_FIELDS,
        OcrConfigKind::VertexAi | OcrConfigKind::VertexDeepSeek => VERTEX_AUTH_OPTION_FIELDS,
        _ => &[],
    };
    Ok(COMMON_OPTION_FIELDS
        .iter()
        .chain(auth_fields)
        .copied()
        .collect())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::mistral("mistral/model", None, &["req_format", "max_response_bytes"], &["vertex_project", "client_secret", "pages"])]
    #[case::vertex("vertex_ai/deepseek-ocr", None, &["vertex_credentials", "vertex_project"], &["client_secret", "aws_region_name"])]
    #[case::azure("model", Some("azure_ai"), &["client_secret", "tenant_id"], &["vertex_credentials"])]
    #[case::textract("aws_textract/detect-document-text", None, &["aws_region_name", "aws_secret_access_key"], &["tenant_id"])]
    fn each_provider_consumes_only_its_own_credentials(
        #[case] model: &str,
        #[case] provider: Option<&str>,
        #[case] consumed: &[&str],
        #[case] not_consumed: &[&str],
    ) {
        let names = owned_option_names(model, provider).unwrap();
        assert!(
            consumed.iter().all(|name| names.contains(name)),
            "{names:?}"
        );
        assert!(
            not_consumed.iter().all(|name| !names.contains(name)),
            "{names:?}"
        );
    }
}
