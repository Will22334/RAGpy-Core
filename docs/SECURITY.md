Security Policy

This document describes the security policy for RAGpy and explains how to report vulnerabilities, how security issues are handled, and what contributors should expect during the review process.

Supported Versions
Security fixes are applied to the most recent stable release of RAGpy. Older versions may not receive patches. Users are encouraged to stay up to date to ensure they receive the latest security improvements.

Reporting a Vulnerability
If you discover a security vulnerability, report it privately to the project maintainers. Do not open a public GitHub issue. Provide a clear description of the problem, steps to reproduce it, and any relevant logs or proof of concept. Responsible disclosure helps protect users and prevents exploitation.

Response Process
Maintainers will acknowledge receipt of your report as soon as possible. The issue will be investigated, and if confirmed, a fix will be developed. You may be contacted for additional information. Once resolved, a patch will be released and the vulnerability will be documented appropriately.

Disclosure Guidelines
Do not disclose the vulnerability publicly until maintainers have released a fix. Premature disclosure may put users at risk. After a patch is available, maintainers may publish details about the issue and credit the reporter unless anonymity is requested.

Security Expectations for Contributors
Contributors should avoid introducing insecure patterns, including hard‑coded secrets, unsafe file handling, insecure network operations, or unvalidated input. Code changes should be reviewed with security in mind. If a contributor identifies a potential issue during development, they should notify maintainers privately.

Dependencies
RAGpy relies on external libraries. Maintainers monitor dependency updates and apply patches when security issues are discovered. Users should update dependencies regularly to reduce exposure to known vulnerabilities.

Contact
To report a vulnerability or raise a security concern, contact the project maintainers through the private communication channel listed in the repository. All reports will be handled confidentially.