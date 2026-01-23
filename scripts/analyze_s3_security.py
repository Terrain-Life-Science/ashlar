#!/usr/bin/env python3
"""
S3 Security Analysis Tool

Analyzes S3 buckets for security issues, checks CloudFront configuration,
and examines bucket contents to understand public access patterns.

Usage:
    python scripts/analyze_s3_security.py [--bucket BUCKET_NAME] [--check-cloudfront]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    print("ERROR: boto3 is required. Install with: pip install boto3")
    sys.exit(1)


class S3SecurityAnalyzer:
    """Analyze S3 buckets for security and CloudFront configuration."""

    def __init__(self):
        """Initialize S3 and CloudFront clients."""
        try:
            self.s3 = boto3.client("s3")
            self.cloudfront = boto3.client("cloudfront")
        except Exception as e:
            print(f"ERROR: Failed to initialize AWS clients: {e}")
            sys.exit(1)

    def check_cloudfront_for_bucket(self, bucket_name: str) -> Dict[str, Any]:
        """
        Check if an S3 bucket is behind CloudFront.

        Parameters
        ----------
        bucket_name : str
            S3 bucket name

        Returns
        -------
        dict
            CloudFront configuration information
        """
        result = {
            "bucket": bucket_name,
            "behind_cloudfront": False,
            "distributions": [],
            "recommendation": "",
        }

        try:
            # List all CloudFront distributions
            distributions = self.cloudfront.list_distributions()

            for dist in distributions.get("DistributionList", {}).get("Items", []):
                # Check origins for this bucket
                for origin in dist.get("Origins", {}).get("Items", []):
                    origin_domain = origin.get("DomainName", "")

                    # S3 bucket origins can be: bucket.s3.region.amazonaws.com or s3-website-region.amazonaws.com
                    if bucket_name in origin_domain or f"{bucket_name}.s3" in origin_domain:
                        dist_info = {
                            "distribution_id": dist.get("Id"),
                            "domain_name": dist.get("DomainName"),
                            "status": dist.get("Status"),
                            "enabled": dist.get("Enabled"),
                            "aliases": dist.get("Aliases", {}).get("Items", []),
                            "origin_path": origin.get("OriginPath", ""),
                        }
                        result["distributions"].append(dist_info)
                        result["behind_cloudfront"] = True

            if result["behind_cloudfront"]:
                result["recommendation"] = (
                    "Bucket is behind CloudFront. Public access may be intentional for website hosting. "
                    "Ensure CloudFront has proper access controls."
                )
            else:
                result["recommendation"] = (
                    "Bucket is NOT behind CloudFront. Direct public access is risky. "
                    "Consider using CloudFront or restricting access."
                )

        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDenied":
                result["recommendation"] = (
                    "Access denied to CloudFront. May require additional permissions."
                )
            else:
                result["recommendation"] = f"Error checking CloudFront: {e}"

        return result

    def analyze_bucket_policy(self, bucket_name: str) -> Dict[str, Any]:
        """
        Analyze S3 bucket policy for public access patterns.

        Parameters
        ----------
        bucket_name : str
            S3 bucket name

        Returns
        -------
        dict
            Policy analysis results
        """
        result = {
            "bucket": bucket_name,
            "has_policy": False,
            "public_access": False,
            "public_principals": [],
            "allowed_actions": [],
            "policy_summary": "",
        }

        try:
            policy = self.s3.get_bucket_policy(Bucket=bucket_name)
            policy_doc = json.loads(policy.get("Policy", "{}"))
            result["has_policy"] = True

            # Analyze statements
            for statement in policy_doc.get("Statement", []):
                principal = statement.get("Principal", {})
                effect = statement.get("Effect", "Deny")
                actions = statement.get("Action", [])

                # Check for public access
                if principal == "*" or (
                    isinstance(principal, dict) and principal.get("AWS") == "*"
                ):
                    result["public_access"] = True
                    result["public_principals"].append("* (Everyone)")
                    result["allowed_actions"].extend(
                        actions if isinstance(actions, list) else [actions]
                    )

                # Check for specific public access patterns
                if isinstance(principal, dict):
                    if principal.get("Service") == "*":
                        result["public_access"] = True
                        result["public_principals"].append("All AWS Services")

            if result["public_access"]:
                result["policy_summary"] = (
                    f"Bucket policy allows public access. Actions: {', '.join(set(result['allowed_actions']))}"
                )
            else:
                result["policy_summary"] = "Bucket policy does not allow public access"

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucketPolicy":
                result["policy_summary"] = "No bucket policy configured"
            else:
                result["policy_summary"] = f"Error reading policy: {e}"

        return result

    def analyze_bucket_contents(self, bucket_name: str, max_files: int = 100) -> Dict[str, Any]:
        """
        Analyze S3 bucket contents to understand what's stored.

        Parameters
        ----------
        bucket_name : str
            S3 bucket name
        max_files : int
            Maximum number of files to analyze (default: 100)

        Returns
        -------
        dict
            Content analysis results
        """
        result = {
            "bucket": bucket_name,
            "total_files": 0,
            "total_size_bytes": 0,
            "file_types": {},
            "sample_files": [],
            "has_sensitive_patterns": False,
            "sensitive_files": [],
            "analysis": "",
        }

        # Patterns that might indicate sensitive data
        sensitive_patterns = [
            ".key",
            ".pem",
            ".p12",
            ".pfx",  # Keys/certificates
            ".env",
            ".config",
            ".conf",  # Config files
            ".sql",
            ".db",
            ".sqlite",  # Databases
            "password",
            "secret",
            "credential",
            "token",  # Keywords
            ".log",
            ".txt",  # May contain sensitive info
        ]

        try:
            paginator = self.s3.get_paginator("list_objects_v2")
            file_count = 0

            for page in paginator.paginate(Bucket=bucket_name, MaxKeys=1000):
                for obj in page.get("Contents", []):
                    file_count += 1
                    key = obj.get("Key", "")
                    size = obj.get("Size", 0)

                    result["total_size_bytes"] += size

                    # Analyze file type
                    if "." in key:
                        ext = key.split(".")[-1].lower()
                        result["file_types"][ext] = result["file_types"].get(ext, 0) + 1

                    # Check for sensitive patterns
                    key_lower = key.lower()
                    for pattern in sensitive_patterns:
                        if pattern in key_lower:
                            result["has_sensitive_patterns"] = True
                            result["sensitive_files"].append(key)
                            break

                    # Collect sample files
                    if len(result["sample_files"]) < 20:
                        result["sample_files"].append(
                            {
                                "key": key,
                                "size": size,
                                "last_modified": (
                                    obj.get("LastModified", "").isoformat()
                                    if obj.get("LastModified")
                                    else None
                                ),
                            }
                        )

                    if file_count >= max_files:
                        break

                if file_count >= max_files:
                    break

            result["total_files"] = file_count

            # Generate analysis
            if result["has_sensitive_patterns"]:
                result["analysis"] = (
                    f"WARNING: Found {len(result['sensitive_files'])} files with potentially sensitive patterns. "
                    f"Review these files before allowing public access."
                )
            elif result["total_files"] == 0:
                result["analysis"] = "Bucket appears to be empty or inaccessible."
            else:
                # Determine bucket purpose from file types
                file_types = result["file_types"]
                if "html" in file_types or "css" in file_types or "js" in file_types:
                    result["analysis"] = (
                        "Bucket appears to contain website files (HTML/CSS/JS). Public access may be intentional."
                    )
                elif "pdf" in file_types or "doc" in file_types or "docx" in file_types:
                    result["analysis"] = (
                        "Bucket appears to contain documents. Verify if public access is needed."
                    )
                elif "jpg" in file_types or "png" in file_types or "gif" in file_types:
                    result["analysis"] = (
                        "Bucket appears to contain images. Public access may be for media hosting."
                    )
                else:
                    result["analysis"] = (
                        f'Bucket contains {result["total_files"]} files. Review contents to determine if public access is appropriate.'
                    )

        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDenied":
                result["analysis"] = (
                    "Access denied to bucket contents. May require additional permissions."
                )
            else:
                result["analysis"] = f"Error analyzing bucket contents: {e}"

        return result

    def analyze_bucket_security(
        self, bucket_name: str, check_cloudfront: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive security analysis of an S3 bucket.

        Parameters
        ----------
        bucket_name : str
            S3 bucket name
        check_cloudfront : bool
            Whether to check CloudFront configuration

        Returns
        -------
        dict
            Complete security analysis
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING BUCKET: {bucket_name}")
        print(f"{'='*60}")

        analysis = {
            "bucket": bucket_name,
            "timestamp": datetime.now().isoformat(),
            "policy_analysis": {},
            "content_analysis": {},
            "cloudfront_analysis": {},
            "recommendations": [],
            "risk_level": "UNKNOWN",
        }

        # Analyze bucket policy
        print("\n[1/3] Analyzing bucket policy...")
        policy_analysis = self.analyze_bucket_policy(bucket_name)
        analysis["policy_analysis"] = policy_analysis
        print(f"  Has policy: {policy_analysis['has_policy']}")
        print(f"  Public access: {policy_analysis['public_access']}")
        print(f"  Summary: {policy_analysis['policy_summary']}")

        # Analyze bucket contents
        print("\n[2/3] Analyzing bucket contents...")
        content_analysis = self.analyze_bucket_contents(bucket_name)
        analysis["content_analysis"] = content_analysis
        print(f"  Total files: {content_analysis['total_files']}")
        print(f"  Total size: {content_analysis['total_size_bytes'] / (1024**2):.2f} MB")
        print(f"  File types: {list(content_analysis['file_types'].keys())[:10]}")
        print(f"  Analysis: {content_analysis['analysis']}")

        # Check CloudFront
        if check_cloudfront:
            print("\n[3/3] Checking CloudFront configuration...")
            cloudfront_analysis = self.check_cloudfront_for_bucket(bucket_name)
            analysis["cloudfront_analysis"] = cloudfront_analysis
            print(f"  Behind CloudFront: {cloudfront_analysis['behind_cloudfront']}")
            if cloudfront_analysis["distributions"]:
                for dist in cloudfront_analysis["distributions"]:
                    print(f"  Distribution: {dist['domain_name']} (Status: {dist['status']})")
            print(f"  Recommendation: {cloudfront_analysis['recommendation']}")

        # Determine risk level and generate recommendations
        if policy_analysis["public_access"]:
            if cloudfront_analysis.get("behind_cloudfront"):
                analysis["risk_level"] = "MEDIUM"
                analysis["recommendations"].append(
                    "Bucket is public but behind CloudFront. Verify CloudFront access controls are properly configured."
                )
            else:
                analysis["risk_level"] = "HIGH"
                analysis["recommendations"].append(
                    "Bucket has direct public access without CloudFront. This is a security risk."
                )
                analysis["recommendations"].append(
                    "Consider: (1) Using CloudFront with proper access controls, (2) Restricting to specific IPs, "
                    "(3) Using signed URLs for temporary access"
                )

        if content_analysis.get("has_sensitive_patterns"):
            analysis["risk_level"] = "HIGH"
            analysis["recommendations"].append(
                f"Found {len(content_analysis['sensitive_files'])} files with sensitive patterns. "
                "Review these files immediately."
            )

        return analysis

    def analyze_all_public_buckets(self, check_cloudfront: bool = True) -> Dict[str, Any]:
        """
        Analyze all S3 buckets that have public access policies.

        Parameters
        ----------
        check_cloudfront : bool
            Whether to check CloudFront configuration

        Returns
        -------
        dict
            Analysis of all public buckets
        """
        print("=" * 60)
        print("ANALYZING ALL S3 BUCKETS FOR PUBLIC ACCESS")
        print("=" * 60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "buckets_analyzed": [],
            "public_buckets": [],
            "summary": {},
        }

        try:
            # List all buckets
            buckets = self.s3.list_buckets()
            bucket_list = buckets.get("Buckets", [])

            print(f"\nFound {len(bucket_list)} bucket(s) to analyze...")

            for bucket in bucket_list:
                bucket_name = bucket["Name"]
                print(f"\n{'='*60}")
                print(f"Analyzing: {bucket_name}")

                # Quick policy check
                policy_analysis = self.analyze_bucket_policy(bucket_name)

                if policy_analysis["public_access"]:
                    # Full analysis for public buckets
                    full_analysis = self.analyze_bucket_security(bucket_name, check_cloudfront)
                    results["public_buckets"].append(full_analysis)
                else:
                    results["buckets_analyzed"].append(
                        {"bucket": bucket_name, "public_access": False}
                    )

            # Generate summary
            results["summary"] = {
                "total_buckets": len(bucket_list),
                "public_buckets_count": len(results["public_buckets"]),
                "high_risk_buckets": len(
                    [b for b in results["public_buckets"] if b["risk_level"] == "HIGH"]
                ),
                "medium_risk_buckets": len(
                    [b for b in results["public_buckets"] if b["risk_level"] == "MEDIUM"]
                ),
            }

        except ClientError as e:
            print(f"ERROR: {e}")
            results["error"] = str(e)

        return results


def main():
    """Main entry point for S3 security analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze S3 buckets for security issues and CloudFront configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific bucket
  python scripts/analyze_s3_security.py --bucket my-bucket
  
  # Analyze all public buckets
  python scripts/analyze_s3_security.py --all
  
  # Skip CloudFront check
  python scripts/analyze_s3_security.py --all --no-cloudfront
        """,
    )

    parser.add_argument("--bucket", type=str, help="Specific bucket name to analyze")

    parser.add_argument("--all", action="store_true", help="Analyze all buckets with public access")

    parser.add_argument(
        "--check-cloudfront",
        action="store_true",
        default=True,
        help="Check CloudFront configuration (default: True)",
    )

    parser.add_argument("--no-cloudfront", action="store_true", help="Skip CloudFront check")

    parser.add_argument("--output", type=Path, help="Output file for JSON results (optional)")

    args = parser.parse_args()

    if args.no_cloudfront:
        args.check_cloudfront = False

    analyzer = S3SecurityAnalyzer()

    try:
        if args.bucket:
            # Analyze specific bucket
            results = analyzer.analyze_bucket_security(args.bucket, args.check_cloudfront)

            # Print summary
            print("\n" + "=" * 60)
            print("ANALYSIS SUMMARY")
            print("=" * 60)
            print(f"Bucket: {results['bucket']}")
            print(f"Risk Level: {results['risk_level']}")
            print("\nRecommendations:")
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"  {i}. {rec}")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n[OK] Results saved to: {args.output}")

        elif args.all:
            # Analyze all public buckets
            results = analyzer.analyze_all_public_buckets(args.check_cloudfront)

            # Print summary
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Total buckets: {results['summary']['total_buckets']}")
            print(f"Public buckets: {results['summary']['public_buckets_count']}")
            print(f"High risk: {results['summary']['high_risk_buckets']}")
            print(f"Medium risk: {results['summary']['medium_risk_buckets']}")

            if results["public_buckets"]:
                print("\nPublic Buckets:")
                for bucket_analysis in results["public_buckets"]:
                    print(f"  - {bucket_analysis['bucket']}: {bucket_analysis['risk_level']} risk")
                    if bucket_analysis["cloudfront_analysis"].get("behind_cloudfront"):
                        print("    (Behind CloudFront)")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n[OK] Results saved to: {args.output}")

        else:
            parser.error("Must specify --bucket or --all")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Analysis failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
