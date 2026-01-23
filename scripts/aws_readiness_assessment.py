#!/usr/bin/env python3
"""
AWS Readiness Assessment Tool

Comprehensive assessment of AWS account readiness for Terrain Life Science
and IAM user jlee8usa. Evaluates account configuration, IAM security,
service availability, resource usage, cost optimization, and best practices.

Usage:
    python scripts/aws_readiness_assessment.py [--output-dir OUTPUT_DIR] [--format json|markdown]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    print("ERROR: boto3 is required. Install with: pip install boto3")
    sys.exit(1)


class AWSReadinessAssessment:
    """Comprehensive AWS account readiness assessment."""

    def __init__(self, account_id: Optional[str] = None, user_name: str = "jlee8usa"):
        """
        Initialize AWS readiness assessment.

        Parameters
        ----------
        account_id : str, optional
            AWS account ID to verify (default: detected from credentials)
        user_name : str
            IAM user name to assess (default: 'jlee8usa')
        """
        self.account_id = account_id
        self.user_name = user_name
        self.results = {
            "assessment_date": datetime.now().isoformat(),
            "account_info": {},
            "iam_assessment": {},
            "ec2_inventory": {},
            "s3_inventory": {},
            "networking": {},
            "monitoring": {},
            "cost_analysis": {},
            "recommendations": [],
            "risks": [],
            "warnings": [],
        }

        # Initialize AWS clients
        self._init_clients()

    def _init_clients(self):
        """Initialize AWS service clients."""
        try:
            # Test basic connection first
            self.sts = boto3.client("sts")
            # Verify we can actually connect
            self.sts.get_caller_identity()
        except Exception as e:
            error_msg = str(e)
            print(f"\nERROR: Failed to initialize AWS clients: {e}\n")

            # Check for specific SSO/login credential provider error
            if "crt" in error_msg.lower() or "login credential provider" in error_msg.lower():
                print("=" * 60)
                print("AWS SSO/LOGIN CREDENTIAL PROVIDER DETECTED")
                print("=" * 60)
                print(
                    "\nYou're using AWS CLI v2 with SSO login, which requires an additional dependency."
                )
                print("\nOption 1: Install the required dependency (Recommended)")
                print("  pip install 'botocore[crt]'")
                print("\nOption 2: Use standard access keys instead of SSO")
                print("  aws configure")
                print("  (Enter Access Key ID and Secret Access Key)")
                print("\nOption 3: Export credentials from SSO session")
                print("  aws sso login")
                print("  # Then export credentials as environment variables")
                print("=" * 60)
            else:
                print("=" * 60)
                print("AWS CREDENTIALS CONFIGURATION")
                print("=" * 60)
                print("\nMake sure AWS credentials are configured:")
                print("  1. Run: aws configure")
                print("  2. Enter your Access Key ID")
                print("  3. Enter your Secret Access Key")
                print("  4. Enter default region (e.g., us-east-1)")
                print("  5. Enter output format (json)")
                print("\nOr if using SSO:")
                print("  1. Run: aws sso login")
                print("  2. Install: pip install 'botocore[crt]'")
                print("=" * 60)

            sys.exit(1)

        # Initialize other clients after successful STS connection
        try:
            self.iam = boto3.client("iam")
            self.ec2 = boto3.client("ec2")
            self.s3 = boto3.client("s3")
            self.cloudwatch = boto3.client("cloudwatch")
            self.cloudtrail = boto3.client("cloudtrail")
            self.service_quotas = boto3.client("service-quotas")
            # Additional clients for comprehensive assessment
            try:
                self.organizations = boto3.client("organizations")
            except:
                self.organizations = None
            try:
                self.costexplorer = boto3.client("ce")
            except:
                self.costexplorer = None
        except Exception as e:
            print(f"WARNING: Some AWS clients failed to initialize: {e}")
            print("Assessment will continue with available clients...")

    def assess_account_info(self) -> Dict[str, Any]:
        """
        Gather account-level information.

        Returns
        -------
        dict
            Account information including account ID, caller identity, and service quotas
        """
        print("=" * 60)
        print("ASSESSING ACCOUNT INFORMATION")
        print("=" * 60)

        account_info = {
            "caller_identity": {},
            "account_status": {},
            "service_quotas": {},
            "regions": [],
        }

        try:
            # Get caller identity
            print("\n[1/4] Getting caller identity...")
            identity = self.sts.get_caller_identity()
            account_info["caller_identity"] = {
                "account_id": identity.get("Account"),
                "user_id": identity.get("UserId"),
                "arn": identity.get("Arn"),
                "is_root": ":root" in identity.get("Arn", ""),
            }

            # Verify account ID matches expected
            if self.account_id and identity.get("Account") != self.account_id:
                self.results["warnings"].append(
                    f"Account ID mismatch: Expected {self.account_id}, got {identity.get('Account')}"
                )

            print(f"  Account ID: {identity.get('Account')}")
            print(f"  ARN: {identity.get('Arn')}")
            print(f"  Is Root: {account_info['caller_identity']['is_root']}")

            if account_info["caller_identity"]["is_root"]:
                self.results["risks"].append(
                    {
                        "severity": "HIGH",
                        "category": "Security",
                        "issue": "Using root account credentials",
                        "recommendation": "Switch to IAM user with appropriate permissions",
                    }
                )

        except ClientError as e:
            print(f"  ERROR: {e}")
            account_info["caller_identity"] = {"error": str(e)}

        try:
            # Get available regions
            print("\n[2/4] Getting available regions...")
            regions_response = self.ec2.describe_regions()
            account_info["regions"] = [r["RegionName"] for r in regions_response.get("Regions", [])]
            print(f"  Found {len(account_info['regions'])} regions")

        except ClientError as e:
            print(f"  ERROR: {e}")
            account_info["regions"] = []

        try:
            # Get service quotas (key services for ashlar_evos)
            print("\n[3/4] Checking service quotas...")
            quotas_to_check = [
                ("ec2", "L-0263D0A3"),  # Running On-Demand EC2 instances
                ("ec2", "L-DB2E81BA"),  # All G and VT Spot Instance Requests
                ("s3", "L-DC2B2D3D"),  # Buckets per account
            ]

            quotas = {}
            for service_code, quota_code in quotas_to_check:
                try:
                    quota = self.service_quotas.get_service_quota(
                        ServiceCode=service_code, QuotaCode=quota_code
                    )
                    quotas[quota_code] = {
                        "service": service_code,
                        "quota_name": quota["Quota"]["QuotaName"],
                        "value": quota["Quota"]["Value"],
                        "adjustable": quota["Quota"]["Adjustable"],
                    }
                except ClientError:
                    # Quota might not be accessible or doesn't exist
                    pass

            account_info["service_quotas"] = quotas
            print(f"  Retrieved {len(quotas)} quota information")

        except ClientError as e:
            print(f"  ERROR accessing service quotas: {e}")
            account_info["service_quotas"] = {}

        try:
            # Get account attributes
            print("\n[4/4] Getting account attributes...")
            # Note: Some account info requires specific permissions
            account_info["account_status"] = {
                "note": "Additional account attributes require specific IAM permissions"
            }

        except Exception as e:
            print(f"  ERROR: {e}")

        self.results["account_info"] = account_info
        print("\n✓ Account information assessment complete")
        return account_info

    def assess_iam(self) -> Dict[str, Any]:
        """
        Assess IAM user security and permissions.

        Returns
        -------
        dict
            IAM assessment including user info, policies, MFA, and access keys
        """
        print("\n" + "=" * 60)
        print("ASSESSING IAM SECURITY")
        print("=" * 60)

        iam_assessment = {
            "user_info": {},
            "policies": {},
            "mfa_enabled": False,
            "access_keys": [],
            "security_issues": [],
        }

        try:
            # Get user information
            print(f"\n[1/5] Getting IAM user information for: {self.user_name}...")
            try:
                user_response = self.iam.get_user(UserName=self.user_name)
                user = user_response["User"]
                iam_assessment["user_info"] = {
                    "user_name": user.get("UserName"),
                    "user_id": user.get("UserId"),
                    "arn": user.get("Arn"),
                    "create_date": (
                        user.get("CreateDate").isoformat() if user.get("CreateDate") else None
                    ),
                    "password_last_used": (
                        user.get("PasswordLastUsed").isoformat()
                        if user.get("PasswordLastUsed")
                        else None
                    ),
                }
                print(f"  User: {user.get('UserName')}")
                print(f"  Created: {user.get('CreateDate')}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchEntity":
                    print(f"  WARNING: User {self.user_name} not found")
                    iam_assessment["user_info"] = {"error": f"User {self.user_name} not found"}
                else:
                    raise

            # Check MFA devices
            print("\n[2/5] Checking MFA devices...")
            try:
                mfa_devices = self.iam.list_mfa_devices(UserName=self.user_name)
                iam_assessment["mfa_enabled"] = len(mfa_devices.get("MFADevices", [])) > 0
                if not iam_assessment["mfa_enabled"]:
                    self.results["risks"].append(
                        {
                            "severity": "HIGH",
                            "category": "Security",
                            "issue": f"MFA not enabled for user {self.user_name}",
                            "recommendation": "Enable MFA for enhanced security",
                        }
                    )
                print(f"  MFA Enabled: {iam_assessment['mfa_enabled']}")
            except ClientError as e:
                print(f"  ERROR checking MFA: {e}")

            # List access keys
            print("\n[3/5] Checking access keys...")
            try:
                access_keys = self.iam.list_access_keys(UserName=self.user_name)
                for key_metadata in access_keys.get("AccessKeyMetadata", []):
                    key_info = {
                        "access_key_id": key_metadata.get("AccessKeyId"),
                        "status": key_metadata.get("Status"),
                        "create_date": (
                            key_metadata.get("CreateDate").isoformat()
                            if key_metadata.get("CreateDate")
                            else None
                        ),
                    }
                    iam_assessment["access_keys"].append(key_info)

                    # Check for old keys (older than 90 days)
                    if key_metadata.get("CreateDate"):
                        from datetime import timezone

                        create_date = key_metadata["CreateDate"]
                        if create_date.tzinfo:
                            now = datetime.now(create_date.tzinfo)
                        else:
                            now = datetime.now(timezone.utc)
                            create_date = create_date.replace(tzinfo=timezone.utc)
                        days_old = (now - create_date).days
                        if days_old > 90:
                            self.results["warnings"].append(
                                f"Access key {key_metadata.get('AccessKeyId')} is {days_old} days old. Consider rotating."
                            )

                    if key_metadata.get("Status") == "Inactive":
                        self.results["warnings"].append(
                            f"Inactive access key found: {key_metadata.get('AccessKeyId')}"
                        )

                print(f"  Found {len(iam_assessment['access_keys'])} access key(s)")
            except ClientError as e:
                print(f"  ERROR listing access keys: {e}")

            # Get attached policies
            print("\n[4/5] Checking attached policies...")
            try:
                # Managed policies
                attached_policies = self.iam.list_attached_user_policies(UserName=self.user_name)
                managed_policies = []
                for policy in attached_policies.get("AttachedPolicies", []):
                    managed_policies.append(
                        {
                            "policy_name": policy.get("PolicyName"),
                            "policy_arn": policy.get("PolicyArn"),
                        }
                    )

                # Inline policies
                inline_policies = self.iam.list_user_policies(UserName=self.user_name)
                inline_policy_names = inline_policies.get("PolicyNames", [])

                iam_assessment["policies"] = {
                    "managed_policies": managed_policies,
                    "inline_policies": inline_policy_names,
                    "total_count": len(managed_policies) + len(inline_policy_names),
                }
                print(f"  Managed policies: {len(managed_policies)}")
                print(f"  Inline policies: {len(inline_policy_names)}")
            except ClientError as e:
                print(f"  ERROR listing policies: {e}")

            # Check for admin policies
            print("\n[5/5] Checking for excessive permissions...")
            try:
                has_admin = False
                for policy in iam_assessment["policies"].get("managed_policies", []):
                    if "AdministratorAccess" in policy.get(
                        "policy_arn", ""
                    ) or "PowerUserAccess" in policy.get("policy_arn", ""):
                        has_admin = True
                        self.results["risks"].append(
                            {
                                "severity": "MEDIUM",
                                "category": "Security",
                                "issue": f"User {self.user_name} has administrative policy attached",
                                "recommendation": "Follow principle of least privilege - use specific permissions",
                            }
                        )

                if not has_admin:
                    print("  No excessive permissions detected")
            except Exception as e:
                print(f"  ERROR checking permissions: {e}")

        except ClientError as e:
            print(f"  ERROR: {e}")
            iam_assessment["error"] = str(e)

        self.results["iam_assessment"] = iam_assessment
        print("\n✓ IAM assessment complete")
        return iam_assessment

    def assess_ec2(self) -> Dict[str, Any]:
        """
        Inventory EC2 resources.

        Returns
        -------
        dict
            EC2 inventory including instances, security groups, VPCs, and key pairs
        """
        print("\n" + "=" * 60)
        print("ASSESSING EC2 RESOURCES")
        print("=" * 60)

        ec2_inventory = {
            "instances": [],
            "security_groups": [],
            "vpcs": [],
            "key_pairs": [],
            "regions_checked": [],
        }

        # Get default region from account info
        default_region = "us-east-1"
        if self.results.get("account_info", {}).get("regions"):
            default_region = (
                self.results["account_info"]["regions"][0]
                if self.results["account_info"]["regions"]
                else "us-east-1"
            )

        regions_to_check = [default_region]  # Start with default region

        try:
            # Get instances
            print(f"\n[1/4] Checking EC2 instances in {default_region}...")
            instances = self.ec2.describe_instances()
            instance_count = 0
            for reservation in instances.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    instance_info = {
                        "instance_id": instance.get("InstanceId"),
                        "instance_type": instance.get("InstanceType"),
                        "state": instance.get("State", {}).get("Name"),
                        "launch_time": (
                            instance.get("LaunchTime").isoformat()
                            if instance.get("LaunchTime")
                            else None
                        ),
                        "tags": {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])},
                        "security_groups": [
                            sg["GroupId"] for sg in instance.get("SecurityGroups", [])
                        ],
                        "vpc_id": instance.get("VpcId"),
                        "subnet_id": instance.get("SubnetId"),
                    }
                    ec2_inventory["instances"].append(instance_info)
                    instance_count += 1

                    # Check for GPU instances (relevant for ashlar_evos)
                    if (
                        "g4dn" in instance.get("InstanceType", "").lower()
                        or "g5" in instance.get("InstanceType", "").lower()
                    ):
                        self.results["recommendations"].append(
                            {
                                "category": "Compute",
                                "finding": f'GPU instance found: {instance.get("InstanceId")} ({instance.get("InstanceType")})',
                                "recommendation": "GPU instances are suitable for ashlar_evos image processing",
                            }
                        )

            print(f"  Found {instance_count} instance(s)")
            ec2_inventory["regions_checked"].append(default_region)

            # Get security groups
            print("\n[2/4] Checking security groups...")
            security_groups = self.ec2.describe_security_groups()
            for sg in security_groups.get("SecurityGroups", []):
                sg_info = {
                    "group_id": sg.get("GroupId"),
                    "group_name": sg.get("GroupName"),
                    "description": sg.get("Description"),
                    "vpc_id": sg.get("VpcId"),
                    "ingress_rules": len(sg.get("IpPermissions", [])),
                    "egress_rules": len(sg.get("IpPermissionsEgress", [])),
                }
                ec2_inventory["security_groups"].append(sg_info)

                # Check for overly permissive rules
                for rule in sg.get("IpPermissions", []):
                    for ip_range in rule.get("IpRanges", []):
                        if ip_range.get("CidrIp") == "0.0.0.0/0":
                            self.results["risks"].append(
                                {
                                    "severity": "MEDIUM",
                                    "category": "Security",
                                    "issue": f'Security group {sg.get("GroupId")} allows traffic from 0.0.0.0/0',
                                    "recommendation": "Restrict source IP ranges to specific IPs or CIDR blocks",
                                }
                            )

            print(f"  Found {len(ec2_inventory['security_groups'])} security group(s)")

            # Get VPCs
            print("\n[3/4] Checking VPCs...")
            vpcs = self.ec2.describe_vpcs()
            for vpc in vpcs.get("Vpcs", []):
                vpc_info = {
                    "vpc_id": vpc.get("VpcId"),
                    "cidr_block": vpc.get("CidrBlock"),
                    "state": vpc.get("State"),
                    "is_default": vpc.get("IsDefault", False),
                    "tags": {tag["Key"]: tag["Value"] for tag in vpc.get("Tags", [])},
                }
                ec2_inventory["vpcs"].append(vpc_info)
            print(f"  Found {len(ec2_inventory['vpcs'])} VPC(s)")

            # Get key pairs
            print("\n[4/4] Checking key pairs...")
            key_pairs = self.ec2.describe_key_pairs()
            for kp in key_pairs.get("KeyPairs", []):
                kp_info = {
                    "key_name": kp.get("KeyName"),
                    "key_fingerprint": kp.get("KeyFingerprint"),
                    "key_type": kp.get("KeyType", "rsa"),
                }
                ec2_inventory["key_pairs"].append(kp_info)
            print(f"  Found {len(ec2_inventory['key_pairs'])} key pair(s)")

        except ClientError as e:
            print(f"  ERROR: {e}")
            ec2_inventory["error"] = str(e)

        self.results["ec2_inventory"] = ec2_inventory
        print("\n✓ EC2 inventory complete")
        return ec2_inventory

    def assess_s3(self) -> Dict[str, Any]:
        """
        Inventory S3 buckets and their security settings.

        Returns
        -------
        dict
            S3 inventory including buckets, policies, encryption, and lifecycle
        """
        print("\n" + "=" * 60)
        print("ASSESSING S3 RESOURCES")
        print("=" * 60)

        s3_inventory = {
            "buckets": [],
            "total_buckets": 0,
            "total_size_bytes": 0,
            "security_issues": [],
        }

        try:
            print("\n[1/3] Listing S3 buckets...")
            buckets = self.s3.list_buckets()
            bucket_list = buckets.get("Buckets", [])
            s3_inventory["total_buckets"] = len(bucket_list)
            print(f"  Found {len(bucket_list)} bucket(s)")

            for bucket in bucket_list:
                bucket_name = bucket["Name"]
                print(f"\n  Analyzing bucket: {bucket_name}")

                bucket_info = {
                    "name": bucket_name,
                    "creation_date": (
                        bucket.get("CreationDate").isoformat()
                        if bucket.get("CreationDate")
                        else None
                    ),
                    "region": None,
                    "encryption": {},
                    "versioning": {},
                    "public_access_block": {},
                    "lifecycle_rules": [],
                    "policy": None,
                }

                try:
                    # Get bucket location
                    try:
                        location = self.s3.get_bucket_location(Bucket=bucket_name)
                        bucket_info["region"] = location.get("LocationConstraint") or "us-east-1"
                    except:
                        bucket_info["region"] = "us-east-1"

                    # Check encryption
                    try:
                        encryption = self.s3.get_bucket_encryption(Bucket=bucket_name)
                        bucket_info["encryption"] = {
                            "enabled": True,
                            "algorithm": encryption.get("ServerSideEncryptionConfiguration", {})
                            .get("Rules", [{}])[0]
                            .get("ApplyServerSideEncryptionByDefault", {})
                            .get("SSEAlgorithm", "Unknown"),
                        }
                    except ClientError as e:
                        if (
                            e.response["Error"]["Code"]
                            != "ServerSideEncryptionConfigurationNotFoundError"
                        ):
                            raise
                        bucket_info["encryption"] = {"enabled": False}
                        self.results["warnings"].append(
                            f"S3 bucket {bucket_name} does not have encryption enabled"
                        )

                    # Check versioning
                    try:
                        versioning = self.s3.get_bucket_versioning(Bucket=bucket_name)
                        bucket_info["versioning"] = {
                            "status": versioning.get("Status", "Disabled"),
                            "mfa_delete": versioning.get("MfaDelete", "Disabled"),
                        }
                    except:
                        bucket_info["versioning"] = {"status": "Unknown"}

                    # Check public access block
                    try:
                        pab = self.s3.get_public_access_block(Bucket=bucket_name)
                        config = pab.get("PublicAccessBlockConfiguration", {})
                        bucket_info["public_access_block"] = {
                            "block_public_acls": config.get("BlockPublicAcls", False),
                            "block_public_policy": config.get("BlockPublicPolicy", False),
                            "ignore_public_acls": config.get("IgnorePublicAcls", False),
                            "restrict_public_buckets": config.get("RestrictPublicBuckets", False),
                        }
                    except ClientError as e:
                        if e.response["Error"]["Code"] != "NoSuchPublicAccessBlockConfiguration":
                            raise
                        bucket_info["public_access_block"] = {"not_configured": True}
                        self.results["risks"].append(
                            {
                                "severity": "HIGH",
                                "category": "Security",
                                "issue": f"S3 bucket {bucket_name} does not have public access block configured",
                                "recommendation": "Enable public access block to prevent accidental public exposure",
                            }
                        )

                    # Check bucket policy
                    try:
                        policy = self.s3.get_bucket_policy(Bucket=bucket_name)
                        bucket_info["policy"] = "configured"
                        # Parse policy to check for public access
                        import json

                        policy_doc = json.loads(policy.get("Policy", "{}"))
                        for statement in policy_doc.get("Statement", []):
                            principal = statement.get("Principal", {})
                            if principal == "*" or (
                                isinstance(principal, dict) and principal.get("AWS") == "*"
                            ):
                                self.results["risks"].append(
                                    {
                                        "severity": "HIGH",
                                        "category": "Security",
                                        "issue": f"S3 bucket {bucket_name} has a policy allowing public access",
                                        "recommendation": "Review and restrict bucket policy to specific principals",
                                    }
                                )
                    except ClientError as e:
                        if e.response["Error"]["Code"] != "NoSuchBucketPolicy":
                            raise
                        bucket_info["policy"] = "none"

                    # Check lifecycle rules
                    try:
                        lifecycle = self.s3.get_bucket_lifecycle_configuration(Bucket=bucket_name)
                        bucket_info["lifecycle_rules"] = [
                            rule.get("Id") for rule in lifecycle.get("Rules", [])
                        ]
                    except ClientError as e:
                        if e.response["Error"]["Code"] != "NoSuchLifecycleConfiguration":
                            raise
                        bucket_info["lifecycle_rules"] = []

                except ClientError as e:
                    print(f"    ERROR analyzing bucket: {e}")
                    bucket_info["error"] = str(e)

                s3_inventory["buckets"].append(bucket_info)

            print("\n[2/3] Bucket analysis complete")
            print("[3/3] Security review complete")

        except ClientError as e:
            print(f"  ERROR: {e}")
            s3_inventory["error"] = str(e)

        self.results["s3_inventory"] = s3_inventory
        print("\n✓ S3 inventory complete")
        return s3_inventory

    def assess_networking(self) -> Dict[str, Any]:
        """
        Assess networking configuration.

        Returns
        -------
        dict
            Networking assessment including VPCs, subnets, security groups, and route tables
        """
        print("\n" + "=" * 60)
        print("ASSESSING NETWORKING")
        print("=" * 60)

        networking = {
            "vpcs": [],
            "subnets": [],
            "route_tables": [],
            "internet_gateways": [],
            "nat_gateways": [],
        }

        try:
            # VPCs (already collected in EC2, but include here for completeness)
            print("\n[1/5] Checking VPCs...")
            vpcs = self.ec2.describe_vpcs()
            networking["vpcs"] = [
                {"vpc_id": vpc["VpcId"], "cidr": vpc.get("CidrBlock")}
                for vpc in vpcs.get("Vpcs", [])
            ]
            print(f"  Found {len(networking['vpcs'])} VPC(s)")

            # Subnets
            print("\n[2/5] Checking subnets...")
            subnets = self.ec2.describe_subnets()
            for subnet in subnets.get("Subnets", []):
                subnet_info = {
                    "subnet_id": subnet.get("SubnetId"),
                    "vpc_id": subnet.get("VpcId"),
                    "cidr_block": subnet.get("CidrBlock"),
                    "availability_zone": subnet.get("AvailabilityZone"),
                    "public": subnet.get("MapPublicIpOnLaunch", False),
                }
                networking["subnets"].append(subnet_info)
            print(f"  Found {len(networking['subnets'])} subnet(s)")

            # Route tables
            print("\n[3/5] Checking route tables...")
            route_tables = self.ec2.describe_route_tables()
            for rt in route_tables.get("RouteTables", []):
                rt_info = {
                    "route_table_id": rt.get("RouteTableId"),
                    "vpc_id": rt.get("VpcId"),
                    "routes": len(rt.get("Routes", [])),
                    "associations": len(rt.get("Associations", [])),
                }
                networking["route_tables"].append(rt_info)
            print(f"  Found {len(networking['route_tables'])} route table(s)")

            # Internet Gateways
            print("\n[4/5] Checking internet gateways...")
            igws = self.ec2.describe_internet_gateways()
            for igw in igws.get("InternetGateways", []):
                igw_info = {
                    "internet_gateway_id": igw.get("InternetGatewayId"),
                    "attached_vpcs": [att.get("VpcId") for att in igw.get("Attachments", [])],
                }
                networking["internet_gateways"].append(igw_info)
            print(f"  Found {len(networking['internet_gateways'])} internet gateway(s)")

            # NAT Gateways
            print("\n[5/5] Checking NAT gateways...")
            nat_gws = self.ec2.describe_nat_gateways()
            for nat in nat_gws.get("NatGateways", []):
                nat_info = {
                    "nat_gateway_id": nat.get("NatGatewayId"),
                    "vpc_id": nat.get("VpcId"),
                    "subnet_id": nat.get("SubnetId"),
                    "state": nat.get("State"),
                }
                networking["nat_gateways"].append(nat_info)
            print(f"  Found {len(networking['nat_gateways'])} NAT gateway(s)")

        except ClientError as e:
            print(f"  ERROR: {e}")
            networking["error"] = str(e)

        self.results["networking"] = networking
        print("\n✓ Networking assessment complete")
        return networking

    def assess_monitoring(self) -> Dict[str, Any]:
        """
        Assess monitoring and logging configuration.

        Returns
        -------
        dict
            Monitoring assessment including CloudWatch, CloudTrail, and AWS Config
        """
        print("\n" + "=" * 60)
        print("ASSESSING MONITORING & LOGGING")
        print("=" * 60)

        monitoring = {"cloudwatch": {}, "cloudtrail": {}, "config": {}}

        try:
            # CloudTrail
            print("\n[1/3] Checking CloudTrail...")
            try:
                trails = self.cloudtrail.describe_trails()
                trail_list = trails.get("trailList", [])
                if trail_list:
                    monitoring["cloudtrail"] = {
                        "trails": [
                            {"name": t.get("Name"), "is_logging": t.get("IsLogging")}
                            for t in trail_list
                        ],
                        "enabled": True,
                    }
                    # Check if logging is enabled
                    for trail in trail_list:
                        if not trail.get("IsLogging"):
                            self.results["risks"].append(
                                {
                                    "severity": "MEDIUM",
                                    "category": "Compliance",
                                    "issue": f"CloudTrail trail {trail.get('Name')} is not logging",
                                    "recommendation": "Enable CloudTrail logging for security auditing",
                                }
                            )
                else:
                    monitoring["cloudtrail"] = {"enabled": False}
                    self.results["risks"].append(
                        {
                            "severity": "HIGH",
                            "category": "Compliance",
                            "issue": "No CloudTrail trails configured",
                            "recommendation": "Enable CloudTrail for API activity logging",
                        }
                    )
                print(f"  Found {len(trail_list)} trail(s)")
            except ClientError as e:
                print(f"  ERROR: {e}")
                monitoring["cloudtrail"] = {"error": str(e)}

            # CloudWatch (basic check)
            print("\n[2/3] Checking CloudWatch...")
            try:
                # List metrics namespaces
                namespaces = self.cloudwatch.list_metrics(MaxResults=1)
                monitoring["cloudwatch"] = {
                    "available": True,
                    "note": "CloudWatch is available (detailed metrics require specific permissions)",
                }
                print("  CloudWatch is available")
            except ClientError as e:
                print(f"  ERROR: {e}")
                monitoring["cloudwatch"] = {"error": str(e)}

            # AWS Config (requires separate client)
            print("\n[3/3] Checking AWS Config...")
            try:
                config_client = boto3.client("config")
                config_recorders = config_client.describe_configuration_recorders()
                recorders = config_recorders.get("ConfigurationRecorders", [])
                if recorders:
                    monitoring["config"] = {"enabled": True, "recorders": len(recorders)}
                else:
                    monitoring["config"] = {"enabled": False}
                    self.results["warnings"].append(
                        "AWS Config is not enabled. Consider enabling for compliance and change tracking."
                    )
                print(f"  AWS Config: {'Enabled' if recorders else 'Not enabled'}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "AccessDenied":
                    monitoring["config"] = {
                        "status": "Access denied - requires additional permissions"
                    }
                else:
                    monitoring["config"] = {"error": str(e)}
                print("  AWS Config: Not accessible (may require additional permissions)")

        except Exception as e:
            print(f"  ERROR: {e}")
            monitoring["error"] = str(e)

        self.results["monitoring"] = monitoring
        print("\n✓ Monitoring assessment complete")
        return monitoring

    def assess_costs(self) -> Dict[str, Any]:
        """
        Assess cost optimization opportunities.

        Returns
        -------
        dict
            Cost analysis including current spending and optimization recommendations
        """
        print("\n" + "=" * 60)
        print("ASSESSING COST OPTIMIZATION")
        print("=" * 60)

        cost_analysis = {"current_month_cost": None, "cost_by_service": {}, "recommendations": []}

        if not self.costexplorer:
            print("\n[1/1] Cost Explorer not available (requires additional permissions)")
            cost_analysis["note"] = "Cost Explorer requires additional IAM permissions"
            self.results["cost_analysis"] = cost_analysis
            return cost_analysis

        try:
            print("\n[1/2] Getting current month costs...")
            # Get costs for current month
            end_date = datetime.now()
            start_date = datetime(end_date.year, end_date.month, 1)

            response = self.costexplorer.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-01"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="MONTHLY",
                Metrics=["BlendedCost"],
            )

            if response.get("ResultsByTime"):
                cost_result = response["ResultsByTime"][0]
                cost_amount = float(
                    cost_result.get("Total", {}).get("BlendedCost", {}).get("Amount", 0)
                )
                cost_analysis["current_month_cost"] = cost_amount
                print(f"  Current month cost: ${cost_amount:.2f}")

            # Get costs by service
            print("\n[2/2] Getting costs by service...")
            service_response = self.costexplorer.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-01"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="MONTHLY",
                Metrics=["BlendedCost"],
                GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
            )

            if service_response.get("ResultsByTime"):
                for group in service_response["ResultsByTime"][0].get("Groups", []):
                    service_name = group.get("Keys", [None])[0]
                    service_cost = float(
                        group.get("Metrics", {}).get("BlendedCost", {}).get("Amount", 0)
                    )
                    if service_cost > 0:
                        cost_analysis["cost_by_service"][service_name] = service_cost

            # Generate recommendations
            if cost_analysis.get("current_month_cost", 0) > 100:
                self.results["recommendations"].append(
                    {
                        "category": "Cost",
                        "finding": f'Monthly costs are ${cost_analysis["current_month_cost"]:.2f}',
                        "recommendation": "Review unused resources and consider Reserved Instances for predictable workloads",
                    }
                )

        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDenied":
                print("  Cost Explorer access denied (requires additional permissions)")
                cost_analysis["note"] = "Cost Explorer requires additional IAM permissions"
            else:
                print(f"  ERROR: {e}")
                cost_analysis["error"] = str(e)

        self.results["cost_analysis"] = cost_analysis
        print("\n✓ Cost analysis complete")
        return cost_analysis

    def generate_markdown_report(self, output_file: Path):
        """
        Generate a markdown report from assessment results.

        Parameters
        ----------
        output_file : Path
            Path to save the markdown report
        """
        with open(output_file, "w") as f:
            f.write("# AWS Readiness Assessment Report\n\n")
            f.write(f"**Assessment Date:** {self.results['assessment_date']}\n\n")
            f.write(
                f"**Account ID:** {self.results['account_info'].get('caller_identity', {}).get('account_id', 'N/A')}\n\n"
            )
            f.write(f"**IAM User:** {self.user_name}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Risks Found:** {len(self.results['risks'])}\n")
            f.write(f"- **Warnings:** {len(self.results['warnings'])}\n")
            f.write(f"- **Recommendations:** {len(self.results['recommendations'])}\n\n")

            # Account Information
            f.write("## Account Information\n\n")
            caller = self.results["account_info"].get("caller_identity", {})
            f.write(f"- **Account ID:** {caller.get('account_id', 'N/A')}\n")
            f.write(f"- **User ARN:** {caller.get('arn', 'N/A')}\n")
            f.write(f"- **Is Root:** {caller.get('is_root', False)}\n")
            f.write(
                f"- **Available Regions:** {len(self.results['account_info'].get('regions', []))}\n\n"
            )

            # IAM Assessment
            f.write("## IAM Security Assessment\n\n")
            iam = self.results.get("iam_assessment", {})
            f.write(f"- **MFA Enabled:** {iam.get('mfa_enabled', False)}\n")
            f.write(f"- **Access Keys:** {len(iam.get('access_keys', []))}\n")
            f.write(f"- **Policies:** {iam.get('policies', {}).get('total_count', 0)}\n\n")

            # EC2 Inventory
            f.write("## EC2 Resources\n\n")
            ec2 = self.results.get("ec2_inventory", {})
            f.write(f"- **Instances:** {len(ec2.get('instances', []))}\n")
            f.write(f"- **Security Groups:** {len(ec2.get('security_groups', []))}\n")
            f.write(f"- **VPCs:** {len(ec2.get('vpcs', []))}\n")
            f.write(f"- **Key Pairs:** {len(ec2.get('key_pairs', []))}\n\n")

            # S3 Inventory
            f.write("## S3 Resources\n\n")
            s3 = self.results.get("s3_inventory", {})
            f.write(f"- **Total Buckets:** {s3.get('total_buckets', 0)}\n")
            buckets_with_encryption = sum(
                1 for b in s3.get("buckets", []) if b.get("encryption", {}).get("enabled", False)
            )
            f.write(f"- **Buckets with Encryption:** {buckets_with_encryption}\n\n")

            # Risks
            if self.results["risks"]:
                f.write("## Security Risks\n\n")
                for i, risk in enumerate(self.results["risks"], 1):
                    f.write(f"### Risk {i}: {risk.get('issue', 'Unknown')}\n\n")
                    f.write(f"- **Severity:** {risk.get('severity', 'Unknown')}\n")
                    f.write(f"- **Category:** {risk.get('category', 'Unknown')}\n")
                    f.write(f"- **Recommendation:** {risk.get('recommendation', 'N/A')}\n\n")

            # Warnings
            if self.results["warnings"]:
                f.write("## Warnings\n\n")
                for warning in self.results["warnings"]:
                    f.write(f"- {warning}\n")
                f.write("\n")

            # Recommendations
            if self.results["recommendations"]:
                f.write("## Recommendations\n\n")
                for rec in self.results["recommendations"]:
                    f.write(
                        f"- **{rec.get('category', 'General')}:** {rec.get('finding', 'N/A')}\n"
                    )
                    f.write(f"  - {rec.get('recommendation', 'N/A')}\n\n")

            f.write("---\n\n")
            f.write("*Report generated by AWS Readiness Assessment Tool*\n")

    def run_full_assessment(self) -> Dict[str, Any]:
        """
        Run complete AWS readiness assessment.

        Returns
        -------
        dict
            Complete assessment results
        """
        print("\n" + "=" * 60)
        print("AWS READINESS ASSESSMENT")
        print("=" * 60)
        print(f"Assessment Date: {self.results['assessment_date']}")
        print(f"Target Account: {self.account_id or 'Auto-detect'}")
        print(f"Target User: {self.user_name}")
        print("=" * 60)

        # Step 1: Account Information
        self.assess_account_info()

        # Step 2: IAM Assessment
        self.assess_iam()

        # Step 3: EC2 Inventory
        self.assess_ec2()

        # Step 4: S3 Inventory
        self.assess_s3()

        # Step 5: Networking Assessment
        self.assess_networking()

        # Step 6: Monitoring Assessment
        self.assess_monitoring()

        # Step 7: Cost Analysis
        self.assess_costs()

        return self.results

    def save_results(self, output_dir: Path, format: str = "json"):
        """
        Save assessment results to file.

        Parameters
        ----------
        output_dir : Path
            Directory to save results
        format : str
            Output format: 'json' or 'markdown'
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "json":
            output_file = output_dir / "aws_readiness_assessment.json"
            with open(output_file, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n✓ Results saved to: {output_file}")
        elif format == "markdown":
            output_file = output_dir / "aws_readiness_assessment.md"
            self.generate_markdown_report(output_file)
            print(f"\n✓ Markdown report saved to: {output_file}")
        else:
            print(f"ERROR: Unknown format: {format}")


def main():
    """Main entry point for AWS readiness assessment."""
    parser = argparse.ArgumentParser(
        description="AWS Readiness Assessment for Terrain Life Science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic assessment
  python scripts/aws_readiness_assessment.py
  
  # Save to specific directory
  python scripts/aws_readiness_assessment.py --output-dir reports/
  
  # Generate markdown report
  python scripts/aws_readiness_assessment.py --format markdown
        """,
    )

    parser.add_argument(
        "--account-id", type=str, help="AWS Account ID to verify (default: auto-detect)"
    )

    parser.add_argument(
        "--user-name",
        type=str,
        default="jlee8usa",
        help="IAM user name to assess (default: jlee8usa)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("aws_assessment_reports"),
        help="Directory to save assessment results (default: aws_assessment_reports/)",
    )

    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    # Run assessment
    assessment = AWSReadinessAssessment(account_id=args.account_id, user_name=args.user_name)

    try:
        results = assessment.run_full_assessment()
        assessment.save_results(args.output_dir, args.format)

        # Print summary
        print("\n" + "=" * 60)
        print("ASSESSMENT SUMMARY")
        print("=" * 60)
        print(
            f"Account ID: {results['account_info'].get('caller_identity', {}).get('account_id', 'N/A')}"
        )
        print(f"Risks Found: {len(results['risks'])}")
        print(f"Warnings: {len(results['warnings'])}")
        print(f"Recommendations: {len(results['recommendations'])}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nAssessment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Assessment failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
