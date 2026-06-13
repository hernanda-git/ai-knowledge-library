# 07 — Supply Chain Security for Agents

## 1. Introduction

AI agents are not built in isolation — they depend on a complex supply chain of models, frameworks, plugins, libraries, and infrastructure. Each dependency introduces potential vulnerabilities that can compromise the entire agent system. Supply chain attacks are particularly insidious because they target trusted components, making detection difficult until after damage has occurred.

For agent systems, the supply chain is broader than traditional software. It includes:
- **Base models**: Pre-trained LLMs that may contain backdoors or biases
- **Fine-tuned models**: Domain-specific models from third-party sources
- **Agent frameworks**: LangChain, AutoGen, CrewAI, and their dependencies
- **Plugins and tools**: Community-developed extensions
- **Vector databases**: Embedding stores that may be compromised
- **Training data**: Datasets that may contain poisoned examples
- **Container images**: Base images for agent deployment
- **APIs and services**: Third-party services the agent integrates with

This document provides comprehensive guidance on securing the agent supply chain, from model provenance verification to dependency scanning and runtime integrity monitoring.

## 2. The Agent Supply Chain Landscape

### 2.1 Supply Chain Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Agent Supply Chain                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │ Model   │  │Training  │  │ Fine-    │  │ Agent Framework │  │
│  │Arch.    │──┤Data      │──┤ Tuning   │──┤ (LangChain,     │  │
│  │         │  │          │  │          │  │  AutoGen, etc.) │  │
│  └─────────┘  └──────────┘  └──────────┘  └────────┬────────┘  │
│                                                      │           │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐            │           │
│  │ Plugin  │  │ Vector   │  │ Code     │            │           │
│  │Marketpl.│  │ Database │──│ Depend.  │◄───────────┘           │
│  │         │  │          │  │ (PyPI)   │                        │
│  └─────────┘  └──────────┘  └──────────┘                        │
│                                                      │           │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐            │           │
│  │ Runtime │  │Container │  │ Infra.   │◄───────────┘           │
│  │ Env.    │  │ Image    │  │ (K8s)    │                        │
│  └─────────┘  └──────────┘  └──────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Threat Vectors in the Agent Supply Chain

| Threat Vector | Description | Risk Level |
|--------------|-------------|------------|
| **Model Backdoor** | Model produces malicious outputs when triggered by specific inputs | Critical |
| **Training Data Poisoning** | Adversarial data inserted during training causes misbehavior | Critical |
| **Dependency Confusion** | Attacker publishes malicious package with same name as internal one | High |
| **Plugin Marketplace Malware** | Malicious plugin in community marketplace | High |
| **Compromised Base Image** | Container image with embedded malware | High |
| **Weight Tampering** | Modification of model weights after distribution | High |
| **Typosquatting** | Malicious package with name similar to popular dependency | Medium |
| **Outdated Dependencies** | Known CVEs in agent framework dependencies | Medium |
| **API Key Leakage** | Keys embedded in published models or config files | Medium |

### 2.3 Real-World Supply Chain Incidents

- **PyTorch dependency confusion (2022)**: Malicious package `pytorch` (note the typo) on PyPI installed cryptocurrency miners on developer machines.
- **Codecov breach (2021)**: Attackers modified Codecov's bash uploader script, exfiltrating credentials from CI/CD pipelines — affecting many AI/ML projects.
- **AI model backdoor research**: Academic demonstrations of backdoored models that perform well on benchmarks but fail maliciously on specific inputs.
- **Hugging Face model scanning findings**: Thousands of uploaded models found to contain malware, pickle-based code execution exploits, or suspicious weights.

## 3. Model Supply Chain Security

### 3.1 Model Signing and Verification

Sign models to ensure integrity and provenance:

```python
import hashlib
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

class ModelSigner:
    """Sign and verify AI model artifacts."""

    def __init__(self, private_key_path: Optional[str] = None):
        if private_key_path:
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
        self.public_key = None

    def sign_model(self, model_path: str) -> dict:
        """Sign a model file/directory with ECDSA signature."""
        # Compute model hash
        model_hash = self._compute_model_hash(model_path)

        # Create signature
        signature = self.private_key.sign(
            model_hash.encode(),
            ec.ECDSA(hashes.SHA256()),
        )

        # Create provenance document
        provenance = {
            "model_hash": model_hash,
            "signature": signature.hex(),
            "algorithm": "ECDSA-SHA256",
            "signer": self._get_signer_id(),
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "metadata": self._collect_metadata(model_path),
        }

        # Save provenance
        provenance_path = f"{model_path}.provenance.json"
        with open(provenance_path, "w") as f:
            json.dump(provenance, f, indent=2)

        return provenance

    def verify_model(self, model_path: str,
                      public_key_path: str) -> tuple[bool, str]:
        """Verify a model's signature."""
        # Load provenance
        with open(f"{model_path}.provenance.json") as f:
            provenance = json.load(f)

        # Load public key
        with open(public_key_path, "rb") as f:
            public_key = serialization.load_pem_public_key(f.read())

        # Verify hash
        current_hash = self._compute_model_hash(model_path)
        if current_hash != provenance["model_hash"]:
            return False, "Model hash mismatch"

        # Verify signature
        try:
            public_key.verify(
                bytes.fromhex(provenance["signature"]),
                provenance["model_hash"].encode(),
                ec.ECDSA(hashes.SHA256()),
            )
            return True, "Signature valid"
        except Exception as e:
            return False, f"Signature invalid: {e}"

    def _compute_model_hash(self, model_path: str) -> str:
        """Compute SHA-256 hash of model file or directory."""
        sha256 = hashlib.sha256()
        if os.path.isfile(model_path):
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256.update(chunk)
        elif os.path.isdir(model_path):
            for root, dirs, files in os.walk(model_path):
                for file in sorted(files):
                    filepath = os.path.join(root, file)
                    sha256.update(filepath.encode())
                    with open(filepath, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            sha256.update(chunk)
        return sha256.hexdigest()
```

### 3.2 SLSA for AI Models

SLSA (Supply-chain Levels for Software Artifacts) provides a framework for supply chain integrity:

```python
class ModelSLSABuilder:
    """
    Implements SLSA compliance for AI model building.
    SLSA Levels 1-4 enforced based on maturity.
    """

    def __init__(self, build_config: dict):
        self.config = build_config
        self.slsa_level = 0

    def build_with_provenance(self, model_config: dict) -> dict:
        """Build a model while generating SLSA-compliant provenance."""
        provenance = {
            "builder": {
                "id": f"{self.config['builder_id']}",
                "version": self.config["builder_version"],
            },
            "buildType": "ai_model_build",
            "materials": [],
            "recipe": {
                "type": "model_training",
                "steps": [],
            },
            "metadata": {
                "buildInvocationID": uuid.uuid4().hex,
                "completeness": {
                    "parameters": True,
                    "environment": True,
                    "materials": True,
                },
                "reproducible": False,
            },
        }

        # Record training materials
        provenance["materials"] = [
            {
                "uri": model_config["base_model_uri"],
                "digest": {"sha256": model_config["base_model_hash"]},
            },
            {
                "uri": model_config["dataset_uri"],
                "digest": {"sha256": model_config["dataset_hash"]},
            },
        ]

        # Record training recipe
        provenance["recipe"]["steps"] = [
            {"command": f"train.py --model {model_config['architecture']}"},
            {"command": f"eval.py --checkpoint {model_config['checkpoint_path']}"},
        ]

        # Generate SLSA level based on controls
        self.slsa_level = self._calculate_slsa_level(provenance)

        return {
            "provenance": provenance,
            "slsa_level": self.slsa_level,
        }

    def verify_provenance(self, provenance: dict) -> dict:
        """Verify the provenance of a built model."""
        findings = []

        # Check materials
        if not provenance.get("materials"):
            findings.append("WARNING: No materials recorded")
        else:
            for material in provenance["materials"]:
                if not material.get("digest"):
                    findings.append(f"WARNING: Material {material['uri']} missing digest")

        # Check builder identity
        if not provenance.get("builder", {}).get("id"):
            findings.append("WARNING: Builder identity not recorded")

        # Check completeness
        completeness = provenance.get("metadata", {}).get("completeness", {})
        missing_params = not completeness.get("parameters", False)
        if missing_params:
            findings.append("WARNING: Incomplete provenance (parameters)")

        return {
            "verified": len(findings) == 0,
            "findings": findings,
            "effective_slsa_level": self._calculate_slsa_level(provenance),
        }

    def _calculate_slsa_level(self, provenance: dict) -> int:
        """Calculate SLSA level based on build controls."""
        level = 1  # Base: provenance exists

        # Level 2: Hosted build + signed provenance
        if self.config.get("hosted_build") and provenance.get("builder", {}).get("id"):
            level = 2

        # Level 3: Hardened build + no user-controlled steps
        if level >= 2 and self.config.get("hardened_build"):
            level = 3

        # Level 4: Hermetic + reproducible + two-person review
        if level >= 3 and self.config.get("hermetic") and self.config.get("reproducible"):
            level = 4

        return level
```

### 3.3 Model Weight Integrity Verification

```python
import struct
import numpy as np

class WeightIntegrityVerifier:
    """Verify the integrity of model weight files."""

    def __init__(self):
        self.expected_checksums = {}
        self.weight_statistics = {}

    def register_model_weights(self, model_id: str,
                                 weight_path: str):
        """Register expected weight statistics for a model."""
        stats = self._compute_weight_statistics(weight_path)
        checksum = self._compute_weight_checksum(weight_path)

        self.expected_checksums[model_id] = checksum
        self.weight_statistics[model_id] = stats

        return {"checksum": checksum, "statistics": stats}

    def verify_weights(self, model_id: str,
                        weight_path: str) -> dict:
        """Verify loaded weights match expected values."""
        issues = []

        # Check checksum
        current_checksum = self._compute_weight_checksum(weight_path)
        expected_checksum = self.expected_checksums.get(model_id)
        if expected_checksum and current_checksum != expected_checksum:
            issues.append("Weight checksum mismatch — weights may have been modified")

        # Check statistical properties
        current_stats = self._compute_weight_statistics(weight_path)
        expected_stats = self.weight_statistics.get(model_id)
        if expected_stats:
            for key in ["mean", "std", "min", "max"]:
                if abs(current_stats.get(key, 0) - expected_stats.get(key, 0)) > 0.01:
                    issues.append(f"Weight statistics mismatch for {key}")
                    break

        # Check for anomalies (e.g., all zeros, extreme values)
        if current_stats and current_stats.get("std", 0) < 0.0001:
            issues.append("Weights have near-zero variance — possible corruption")
        if current_stats and abs(current_stats.get("max", 0)) > 100:
            issues.append("Weights contain extreme values — possible poisoning")

        return {
            "verified": len(issues) == 0,
            "issues": issues,
            "current_checksum": current_checksum,
        }

    def _compute_weight_checksum(self, weight_path: str) -> str:
        """Compute checksum of weight file (using first 1MB for speed)."""
        sha256 = hashlib.sha256()
        with open(weight_path, "rb") as f:
            chunk = f.read(1048576)  # First 1MB
            sha256.update(chunk)
        return sha256.hexdigest()

    def _compute_weight_statistics(self, weight_path: str) -> dict:
        """Compute statistical properties of weights."""
        try:
            # Attempt to load as numpy
            weights = np.load(weight_path)
            if isinstance(weights, np.lib.npyio.NpzFile):
                # For .npz files, analyze first array
                arr_name = list(weights.keys())[0]
                data = weights[arr_name].flatten()
            else:
                data = weights.flatten()

            return {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "has_nan": bool(np.any(np.isnan(data))),
                "has_inf": bool(np.any(np.isinf(data))),
                "size_bytes": data.nbytes,
            }
        except Exception:
            return {}
```

### 3.4 Model Registry with Provenance

```python
class ModelRegistry:
    """Register models with provenance tracking."""

    def __init__(self):
        self.models = {}         # model_id -> model metadata
        self.provenance = {}     # model_id -> provenance chain

    def register_model(self, model_id: str, model_path: str,
                        metadata: dict) -> dict:
        """Register a model with its provenance."""
        # Compute hashes
        file_hash = self._hash_file(model_path)
        metadata_hash = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()

        self.models[model_id] = {
            "model_id": model_id,
            "path": model_path,
            "hash": file_hash,
            "metadata_hash": metadata_hash,
            "metadata": metadata,
            "registered_at": datetime.now().isoformat(),
            "status": "active",
        }

        # Initialize provenance chain
        self.provenance[model_id] = [
            {
                "event": "registration",
                "model_id": model_id,
                "hash": file_hash,
                "timestamp": datetime.now().isoformat(),
                "actor": metadata.get("registrar", "unknown"),
            }
        ]

        return self.models[model_id]

    def record_provenance_event(self, model_id: str,
                                  event: str, details: dict):
        """Record a provenance event (deployment, update, audit, etc.)."""
        if model_id not in self.provenance:
            raise ValueError(f"Model {model_id} not registered")

        self.provenance[model_id].append({
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "previous_hash": self._get_latest_hash(model_id),
        })

    def get_provenance_chain(self, model_id: str) -> list[dict]:
        """Get the complete provenance chain for a model."""
        return self.provenance.get(model_id, [])

    def verify_provenance_chain(self, model_id: str) -> tuple[bool, list[str]]:
        """Verify the integrity of the provenance chain."""
        chain = self.provenance.get(model_id, [])
        issues = []

        for i, event in enumerate(chain):
            if i > 0:
                expected_prev = chain[i-1].get("_hash", "")
                actual_prev = event.get("previous_hash", "")
                if expected_prev and actual_prev != expected_prev:
                    issues.append(f"Provenance chain broken at event {i}")

        return len(issues) == 0, issues
```

## 4. Dependency Security for Agent Frameworks

### 4.1 Dependency Scanning

```python
import subprocess
import json
from datetime import datetime

class AgentDependencyScanner:
    """Scan agent framework dependencies for vulnerabilities."""

    def __init__(self):
        self.vulnerability_db = {}
        self.scan_results = {}

    def scan_python_dependencies(self, requirements_path: str) -> dict:
        """Scan Python dependencies using pip-audit or safety."""
        try:
            # Using pip-audit
            result = subprocess.run(
                ["pip-audit", "--requirement", requirements_path,
                 "--format", "json"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 or result.returncode == 1:
                return json.loads(result.stdout)
            else:
                return {"error": result.stderr}
        except FileNotFoundError:
            # Fallback to pip list check
            return self._basic_dependency_check(requirements_path)
        except subprocess.TimeoutExpired:
            return {"error": "Dependency scan timed out"}

    def scan_npm_dependencies(self, package_json_path: str) -> dict:
        """Scan npm dependencies for agent tooling."""
        try:
            result = subprocess.run(
                ["npm", "audit", "--json"],
                cwd=os.path.dirname(package_json_path),
                capture_output=True, text=True, timeout=120,
            )
            return json.loads(result.stdout)
        except Exception as e:
            return {"error": str(e)}

    def check_known_vulnerabilities(self, package_name: str,
                                     version: str) -> list[dict]:
        """Check a specific package version against known CVEs."""
        # In production, query OSV.dev or NVD database
        vulnerabilities = []
        for cve, details in self.vulnerability_db.items():
            if details.get("package") == package_name:
                if self._version_in_range(version, details.get("affected_versions", [])):
                    vulnerabilities.append({
                        "cve": cve,
                        "severity": details.get("severity"),
                        "description": details.get("description"),
                    })
        return vulnerabilities

    def generate_sbom(self, project_path: str) -> dict:
        """Generate a Software Bill of Materials for agent project."""
        try:
            result = subprocess.run(
                ["cyclonedx-py", project_path, "--format", "json"],
                capture_output=True, text=True, timeout=120,
            )
            return json.loads(result.stdout)
        except Exception as e:
            return self._manual_sbom(project_path)

    def _manual_sbom(self, project_path: str) -> dict:
        """Generate basic SBOM manually."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "component": {
                    "name": os.path.basename(project_path),
                    "type": "application",
                }
            },
            "components": [],
        }

        # Parse requirements.txt
        req_file = os.path.join(project_path, "requirements.txt")
        if os.path.exists(req_file):
            with open(req_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split("==")
                        sbom["components"].append({
                            "name": parts[0],
                            "version": parts[1] if len(parts) > 1 else "latest",
                            "type": "library",
                        })

        return sbom
```

### 4.2 Dependency Pinning and Locking

```python
class DependencyLockManager:
    """Manage locked dependencies for reproducible agent builds."""

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.lock_files = {
            "python": "requirements-lock.txt",
            "javascript": "package-lock.json",
            "docker": "Dockerfile.sha256",
        }

    def create_lock_file(self, language: str = "python") -> str:
        """Create a dependency lock file."""
        if language == "python":
            return self._lock_python_deps()
        return ""

    def _lock_python_deps(self) -> str:
        """Create Python dependency lock file with hashes."""
        result = subprocess.run(
            ["pip", "list", "--format", "freeze"],
            capture_output=True, text=True,
        )
        packages = result.stdout.strip().split("\n")

        lock_entries = []
        for package in packages:
            if "==" in package:
                name, version = package.split("==")
                # Get wheel hash
                hash_result = subprocess.run(
                    ["pip", "hash", f"{name}=={version}"],
                    capture_output=True, text=True,
                )
                pkg_hash = hash_result.stdout.strip().split("\n")[0] if hash_result.stdout else ""

                lock_entries.append({
                    "name": name,
                    "version": version,
                    "hash": pkg_hash,
                })

        lock_content = {
            "generated_at": datetime.now().isoformat(),
            "python_version": subprocess.run(
                ["python3", "--version"], capture_output=True, text=True
            ).stdout.strip(),
            "packages": lock_entries,
        }

        lock_path = os.path.join(self.project_path, self.lock_files["python"])
        with open(lock_path, "w") as f:
            json.dump(lock_content, f, indent=2)

        return lock_path

    def verify_lock_file(self, language: str = "python") -> dict:
        """Verify installed packages match lock file."""
        lock_path = os.path.join(self.project_path, self.lock_files.get(language, ""))
        if not os.path.exists(lock_path):
            return {"valid": False, "error": "Lock file not found"}

        with open(lock_path) as f:
            lock_content = json.load(f)

        # Get current packages
        result = subprocess.run(
            ["pip", "list", "--format", "json"],
            capture_output=True, text=True,
        )
        current = json.loads(result.stdout)

        issues = []
        for locked_pkg in lock_content.get("packages", []):
            found = False
            for current_pkg in current:
                if current_pkg["name"] == locked_pkg["name"]:
                    found = True
                    if current_pkg["version"] != locked_pkg["version"]:
                        issues.append(
                            f"Version mismatch: {locked_pkg['name']} "
                            f"(locked: {locked_pkg['version']}, "
                            f"installed: {current_pkg['version']})"
                        )
                    break
            if not found:
                issues.append(f"Missing package: {locked_pkg['name']}")

        return {"valid": len(issues) == 0, "issues": issues}
```

## 5. Plugin and Marketplace Security

### 5.1 Plugin Vetting Pipeline

```python
class PluginVettingPipeline:
    """Security vetting pipeline for agent plugins."""

    def __init__(self):
        self.checks = [
            StaticAnalysisCheck(),
            DependencyScanCheck(),
            PermissionAuditCheck(),
            CodeSigningCheck(),
            BehaviorAnalysisCheck(),
        ]

    def vet_plugin(self, plugin_path: str) -> dict:
        """Run comprehensive security vetting on a plugin."""
        results = {
            "plugin": plugin_path,
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "passed": True,
            "risk_score": 0.0,
        }

        for check in self.checks:
            try:
                check_result = check.run(plugin_path)
                results["checks"].append(check_result)
                if not check_result["passed"]:
                    results["passed"] = False
            except Exception as e:
                results["checks"].append({
                    "name": check.__class__.__name__,
                    "passed": False,
                    "error": str(e),
                })
                results["passed"] = False

        # Calculate risk score
        failed = sum(1 for c in results["checks"] if not c["passed"])
        results["risk_score"] = failed / len(self.checks)

        return results

class StaticAnalysisCheck:
    """Static code analysis for malicious patterns."""
    def run(self, plugin_path: str) -> dict:
        issues = []
        suspicious_patterns = [
            "subprocess.call", "os.system", "exec(", "eval(",
            "__import__", "base64.b64decode", "urllib.request",
            "socket.socket", "pickle.loads", "requests.get(",
        ]

        for root, dirs, files in os.walk(plugin_path):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    with open(filepath) as f:
                        content = f.read()
                        for pattern in suspicious_patterns:
                            if pattern in content:
                                issues.append({
                                    "file": filepath,
                                    "pattern": pattern,
                                    "severity": "medium",
                                })

        return {
            "name": "Static Analysis",
            "passed": len(issues) == 0,
            "issues": issues,
        }

class DependencyScanCheck:
    """Scan plugin dependencies for vulnerabilities."""
    def run(self, plugin_path: str) -> dict:
        req_file = os.path.join(plugin_path, "requirements.txt")
        if not os.path.exists(req_file):
            return {"name": "Dependency Scan", "passed": True, "note": "No dependencies"}

        scanner = AgentDependencyScanner()
        scan_result = scanner.scan_python_dependencies(req_file)
        vulnerabilities = scan_result.get("vulnerabilities", [])

        return {
            "name": "Dependency Scan",
            "passed": len(vulnerabilities) == 0,
            "vulnerabilities": vulnerabilities,
        }
```

### 5.2 Plugin Sandboxing

```python
class PluginSandbox:
    """Sandbox environment for executing untrusted plugins."""

    def __init__(self):
        self.sandbox_dir = tempfile.mkdtemp(prefix="plugin_sandbox_")

    def execute_in_sandbox(self, plugin_func: callable,
                            *args, **kwargs) -> dict:
        """Execute a plugin function in a sandboxed environment."""
        # Set up restricted environment
        original_cwd = os.getcwd()
        original_path = sys.path.copy()

        try:
            os.chdir(self.sandbox_dir)
            sys.path = [self.sandbox_dir]

            # Set resource limits
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))

            # Execute
            result = plugin_func(*args, **kwargs)

            return {"success": True, "result": result}

        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            os.chdir(original_cwd)
            sys.path = original_path
            # Clean up
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
```

### 5.3 Plugin Permission Manifest

```python
class PluginPermissionManifest:
    """
    Declarative permission manifest for agent plugins.
    Similar to Android/iOS app permissions.
    """

    MANIFEST_SCHEMA = {
        "plugin_name": str,
        "version": str,
        "author": str,
        "permissions": {
            "network": {
                "allowed_domains": list,
                "allowed_protocols": list,
            },
            "filesystem": {
                "read_paths": list,
                "write_paths": list,
            },
            "agent": {
                "can_read_memory": bool,
                "can_modify_config": bool,
                "can_access_tools": list,
            },
            "data": {
                "data_types_accessed": list,
                "data_retention_days": int,
            },
        },
        "dependencies": list,
    }

    @staticmethod
    def validate_manifest(manifest_path: str) -> dict:
        """Validate a plugin manifest against the schema."""
        with open(manifest_path) as f:
            manifest = json.load(f)

        issues = []
        warnings = []

        # Check required fields
        for field in ["plugin_name", "version", "author", "permissions"]:
            if field not in manifest:
                issues.append(f"Missing required field: {field}")

        # Check permissions
        permissions = manifest.get("permissions", {})
        if permissions.get("network", {}).get("allowed_domains") == ["*"]:
            warnings.append("Plugin requests unrestricted network access")
        if permissions.get("filesystem", {}).get("read_paths") == ["*"]:
            warnings.append("Plugin requests unrestricted file system read access")
        if permissions.get("agent", {}).get("can_modify_config"):
            warnings.append("Plugin can modify agent configuration")
        if permissions.get("agent", {}).get("can_read_memory"):
            warnings.append("Plugin can access agent conversation memory")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "risk_level": self._calculate_risk(warnings),
        }

    @staticmethod
    def _calculate_risk(warnings: list) -> str:
        risk_weights = {
            "unrestricted network access": 0.4,
            "unrestricted file system": 0.3,
            "modify agent configuration": 0.3,
            "access agent conversation memory": 0.4,
        }
        total_risk = sum(
            risk_weights.get(w.split(": ")[-1], 0.1)
            for w in warnings
        )
        if total_risk > 0.7:
            return "CRITICAL"
        elif total_risk > 0.4:
            return "HIGH"
        elif total_risk > 0.2:
            return "MEDIUM"
        return "LOW"
```

## 6. Container Security for Agent Deployments

### 6.1 Secure Base Images

```yaml
# Dockerfile for secure agent deployment
FROM python:3.11-slim AS base

# Use non-root user
RUN groupadd -r agent && useradd -r -g agent -m -d /app agent

# Install only necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set secure defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy only verified dependencies
COPY --chown=agent:agent requirements-lock.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-lock.txt

# Copy application
COPY --chown=agent:agent . /app/
WORKDIR /app

# Drop all capabilities
USER agent

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

ENTRYPOINT ["python", "agent_main.py"]
```

### 6.2 Container Image Scanning

```python
class ContainerImageScanner:
    """Scan container images for vulnerabilities."""

    def __init__(self, registry_url: str):
        self.registry = registry_url

    def scan_image(self, image_name: str, tag: str = "latest") -> dict:
        """Scan a container image using Trivy or Grype."""
        image_ref = f"{self.registry}/{image_name}:{tag}"

        try:
            # Trivy scan
            result = subprocess.run(
                ["trivy", "image", "--format", "json",
                 "--severity", "CRITICAL,HIGH", image_ref],
                capture_output=True, text=True, timeout=300,
            )
            return json.loads(result.stdout)
        except FileNotFoundError:
            # Fallback to Grype
            try:
                result = subprocess.run(
                    ["grype", image_ref, "--output", "json"],
                    capture_output=True, text=True, timeout=300,
                )
                return json.loads(result.stdout)
            except FileNotFoundError:
                return {"error": "No vulnerability scanner found (install trivy or grype)"}
        except subprocess.TimeoutExpired:
            return {"error": "Image scan timed out"}
        except json.JSONDecodeError:
            return {"error": "Failed to parse scan results"}

    def check_image_integrity(self, image_name: str,
                               tag: str = "latest") -> dict:
        """Check container image signatures."""
        try:
            # Cosign verification
            result = subprocess.run(
                ["cosign", "verify", f"{self.registry}/{image_name}:{tag}"],
                capture_output=True, text=True, timeout=60,
            )
            return {
                "signed": result.returncode == 0,
                "output": result.stdout,
            }
        except FileNotFoundError:
            return {"error": "Cosign not installed"}
```

### 6.3 Runtime Container Security

```python
class AgentContainerRuntime:
    """Secure container runtime configuration for agents."""

    def __init__(self, agent_config: dict):
        self.config = agent_config

    def generate_kubernetes_security_context(self) -> dict:
        """Generate Kubernetes security context for agent pods."""
        return {
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 1001,
                "runAsGroup": 1001,
                "fsGroup": 1001,
                "seccompProfile": {
                    "type": "RuntimeDefault",
                },
                "capabilities": {
                    "drop": ["ALL"],
                },
                "readOnlyRootFilesystem": True,
                "allowPrivilegeEscalation": False,
            },
            "podSecurityContext": {
                "runAsNonRoot": True,
                "seccompProfile": {
                    "type": "RuntimeDefault",
                },
            },
        }

    def generate_network_policy(self) -> dict:
        """Generate network policy for agent pods."""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{self.config['agent_name']}-network-policy",
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": self.config["agent_name"],
                    },
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {"from": [{"ipBlock": {"cidr": self.config.get("allowed_ingress_cidr", "10.0.0.0/8")}}]},
                ],
                "egress": [
                    {
                        "to": [
                            {"ipBlock": {"cidr": self.config.get("allowed_egress_cidr", "10.0.0.0/8")}},
                        ],
                        "ports": [
                            {"protocol": "TCP", "port": 443},
                            {"protocol": "TCP", "port": 80},
                        ],
                    },
                ],
            },
        }
```

## 7. SBOM for AI Systems

### 7.1 AI SBOM Schema

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "metadata": {
    "timestamp": "2026-06-13T12:00:00Z",
    "component": {
      "name": "customer-support-agent-v2",
      "version": "2.1.0",
      "type": "application",
      "ai": {
        "model": {
          "type": "llm",
          "name": "llama-3-70b",
          "version": "3.0",
          "source": "huggingface.co/meta-llama",
          "hash": "sha256:abc123...",
          "architecture": "transformer",
          "parameters": 70000000000,
          "training_data_summary": "Public web data up to 2023-12",
          "fine_tuning": {
            "dataset": "customer-support-v3",
            "dataset_hash": "sha256:def456...",
            "technique": "lora",
            "trainer": "axolotl"
          }
        }
      }
    }
  },
  "components": [
    {"name": "langchain", "version": "0.2.0", "type": "library"},
    {"name": "chromadb", "version": "0.4.22", "type": "library"},
    {"name": "transformers", "version": "4.40.0", "type": "library"}
  ],
  "dependencies": [
    {"ref": "customer-support-agent-v2@2.1.0", "dependsOn": [
      "langchain@0.2.0",
      "chromadb@0.4.22",
      "transformers@4.40.0"
    ]}
  ],
  "vulnerabilities": []
}
```

### 7.2 SBOM Generation for Agent Systems

```python
class AgentSBOMGenerator:
    """Generate Software Bill of Materials for AI agent systems."""

    def __init__(self, agent_name: str, agent_version: str):
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.components = []
        self.dependencies = []

    def add_model_component(self, model_info: dict):
        """Add an AI model to the SBOM."""
        component = {
            "name": model_info["name"],
            "version": model_info["version"],
            "type": "ai-model",
            "ai": {
                "model": {
                    "type": model_info.get("type", "llm"),
                    "source": model_info.get("source", ""),
                    "hash": model_info.get("hash", ""),
                    "architecture": model_info.get("architecture", ""),
                    "parameters": model_info.get("parameters", 0),
                }
            },
            "evidence": {
                "licenses": model_info.get("licenses", []),
                "copyright": model_info.get("copyright", ""),
            },
        }
        self.components.append(component)
        return component

    def add_dataset_component(self, dataset_info: dict):
        """Add a training dataset to the SBOM."""
        component = {
            "name": dataset_info["name"],
            "version": dataset_info.get("version", "1.0"),
            "type": "dataset",
            "description": dataset_info.get("description", ""),
            "properties": [
                {"name": "record_count", "value": str(dataset_info.get("record_count", 0))},
                {"name": "source", "value": dataset_info.get("source", "")},
                {"name": "data_types", "value": ",".join(dataset_info.get("data_types", []))},
            ],
        }
        self.components.append(component)
        return component

    def add_package_component(self, name: str, version: str,
                                purl: str = None):
        """Add a software package to the SBOM."""
        component = {
            "name": name,
            "version": version,
            "type": "library",
            "purl": purl or f"pkg:pypi/{name}@{version}",
        }
        self.components.append(component)
        return component

    def generate(self) -> dict:
        """Generate the complete SBOM."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "tools": [
                    {"vendor": "Agent Knowledge Base", "name": "sbom-generator", "version": "1.0.0"}
                ],
                "component": {
                    "name": self.agent_name,
                    "version": self.agent_version,
                    "type": "application",
                },
            },
            "components": self.components,
            "dependencies": self.dependencies,
        }

        # Generate dependency graph
        root_ref = f"{self.agent_name}@{self.agent_version}"
        sbom["dependencies"].append({
            "ref": root_ref,
            "dependsOn": [c["name"] for c in self.components],
        })

        return sbom

    def export(self, output_path: str):
        """Export SBOM to file."""
        sbom = self.generate()
        with open(output_path, "w") as f:
            json.dump(sbom, f, indent=2)
        return output_path
```

## 8. Framework-Specific Security

### 8.1 LangChain Dependency Verification

```python
class LangChainSecurityCheck:
    """Security checks specific to LangChain deployments."""

    def __init__(self):
        self.known_vulnerabilities = {
            "langchain": {
                "0.1.0": ["CVE-2024-1234: Prompt injection via tool descriptions"],
                "0.1.5": ["CVE-2024-5678: Arbitrary code execution in PythonREPL"],
            },
            "langchain-community": {
                "0.1.0": ["CVE-2024-9012: SSRF in web_search tool"],
            },
        }

    def check_langchain_version(self, version: str) -> list[dict]:
        """Check LangChain version for known vulnerabilities."""
        findings = []
        for pkg, vulns in self.known_vulnerabilities.items():
            if version in vulns:
                for vuln in vulns[version]:
                    findings.append({
                        "package": pkg,
                        "version": version,
                        "vulnerability": vuln,
                    })
        return findings

    def validate_tool_definitions(self, tools: list) -> list[str]:
        """Validate tool definitions for security issues."""
        issues = []
        for tool in tools:
            # Check for dangerous tool names
            dangerous_names = ["exec", "eval", "shell", "bash", "terminal"]
            if any(d in tool.name.lower() for d in dangerous_names):
                issues.append(f"Potentially dangerous tool: {tool.name}")

            # Check for improper parameter schemas
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.schema()
                if 'exec' in str(schema.get('properties', {})).lower():
                    issues.append(f"Tool {tool.name} has suspicious parameter schema")
        return issues
```

### 8.2 AutoGen Supply Chain Security

```python
class AutoGenSecurityValidator:
    """Validate AutoGen agent configurations for supply chain risks."""

    def __init__(self):
        self.trusted_models = {
            "gpt-4-turbo", "gpt-4o", "claude-3-opus",
            "claude-3-sonnet", "llama-3-70b", "llama-3-8b",
        }

    def validate_agent_config(self, config: dict) -> dict:
        """Validate an AutoGen agent configuration."""
        issues = []

        # Check model source
        model = config.get("llm_config", {}).get("model", "")
        if model not in self.trusted_models:
            issues.append(f"Model '{model}' not in trusted model list")

        # Check API key source (should not be hardcoded)
        api_key = config.get("llm_config", {}).get("api_key", "")
        if api_key and not api_key.startswith("$"):
            issues.append("API key appears to be hardcoded (use env variable)")

        # Check for dangerous function definitions
        functions = config.get("functions", [])
        for func in functions:
            if func.get("name") in ("exec_code", "run_shell", "eval_expression"):
                issues.append(f"Dangerous function in agent config: {func['name']}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "agent_id": config.get("name", "unknown"),
        }
```

## 9. Conclusion

Supply chain security for AI agents requires a comprehensive approach that addresses the unique components of the AI stack — models, training data, frameworks, plugins, and infrastructure — alongside traditional software dependencies.

Key implementation priorities:

1. **Sign and verify models**: Use cryptographic signing for all model artifacts.
2. **Maintain provenance**: Track the complete lineage of every model and dataset.
3. **Scan all dependencies**: Regularly scan framework dependencies for vulnerabilities.
4. **Lock dependencies**: Use hash-locked dependency files for reproducible builds.
5. **Vet plugins rigorously**: Run security checks on every plugin before installation.
6. **Generate SBOMs**: Maintain a complete software bill of materials for every agent.
7. **Harden containers**: Use minimal base images, drop capabilities, implement network policies.
8. **Monitor continuously**: Watch for new vulnerabilities in agent dependencies.
9. **Verify at load time**: Check model integrity before loading into memory.
10. **Plan for updates**: Have a process for quickly patching supply chain vulnerabilities.

---

**References**
- SLSA Framework (Supply-chain Levels for Software Artifacts)
- CycloneDX SBOM Standard v1.5
- OpenSSF Scorecard
- Sigstore/Cosign for container signing
- Trivy Vulnerability Scanner
- Hugging Face Model Security Best Practices
- NIST SP 800-161: Supply Chain Risk Management

---

**Document Information**
- Title: Supply Chain Security for Agents
- Series: 18-Agent-Security-and-Trust
- Part: 07 of 08
- Author: AI Knowledge Base
- Last Updated: 2026-06-13
- Lines: 615
