#!/usr/bin/env python3
"""
HPE Training Monitoring Dashboard
Step 5.3: Fine-tune HPE with New Data

Simple web dashboard to monitor HPE training progress on AWS.
"""

import json
import boto3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import argparse

try:
    from flask import Flask, render_template, jsonify
    import plotly.graph_objs as go
    import plotly.utils
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

class HPETrainingMonitor:
    """Monitor HPE training progress and costs"""
    
    def __init__(self, region: str = 'us-east-2'):
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.s3 = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    def get_training_instances(self) -> List[Dict]:
        """Get all HPE training instances"""
        response = self.ec2.describe_instances(
            Filters=[
                {'Name': 'tag:Project', 'Values': ['VLR-HPE-Research']},
                {'Name': 'instance-state-name', 'Values': ['running', 'pending']}
            ]
        )
        
        instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_info = {
                    'instance_id': instance['InstanceId'],
                    'instance_type': instance['InstanceType'],
                    'state': instance['State']['Name'],
                    'launch_time': instance['LaunchTime'],
                    'public_ip': instance.get('PublicIpAddress', 'N/A'),
                    'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                }
                instances.append(instance_info)
        
        return instances
    
    def get_training_metrics(self, instance_id: str) -> Dict:
        """Get CloudWatch metrics for training instance"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        metrics = {}
        
        # CPU Utilization
        try:
            cpu_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )
            metrics['cpu'] = cpu_response['Datapoints']
        except:
            metrics['cpu'] = []
        
        # Network metrics
        try:
            network_in = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='NetworkIn',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Sum']
            )
            metrics['network_in'] = network_in['Datapoints']
        except:
            metrics['network_in'] = []
        
        return metrics
    
    def estimate_costs(self, instances: List[Dict]) -> Dict:
        """Estimate training costs"""
        # On-demand pricing (USD per hour) - approximate values
        pricing = {
            'g4dn.xlarge': 0.526,
            'p3.2xlarge': 3.06,
            'p4d.24xlarge': 32.77,
            'p3.8xlarge': 12.24
        }
        
        total_cost = 0.0
        instance_costs = {}
        
        for instance in instances:
            instance_type = instance['instance_type']
            launch_time = instance['launch_time']
            
            # Calculate uptime hours
            if isinstance(launch_time, str):
                launch_time = datetime.fromisoformat(launch_time.replace('Z', '+00:00'))
            
            uptime_hours = (datetime.now(launch_time.tzinfo) - launch_time).total_seconds() / 3600
            
            # Estimate cost (assuming spot discount of 60% for spot instances)
            hourly_rate = pricing.get(instance_type, 1.0)
            if instance['tags'].get('SpotInstance', 'false').lower() == 'true':
                hourly_rate *= 0.4  # 60% discount for spot
            
            instance_cost = uptime_hours * hourly_rate
            total_cost += instance_cost
            
            instance_costs[instance['instance_id']] = {
                'uptime_hours': uptime_hours,
                'hourly_rate': hourly_rate,
                'total_cost': instance_cost
            }
        
        return {
            'total_cost': total_cost,
            'instance_costs': instance_costs
        }
    
    def check_s3_training_outputs(self, bucket: str, prefix: str = 'hpe_models') -> Dict:
        """Check S3 for training outputs and checkpoints"""
        try:
            response = self.s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            files = []
            total_size = 0
            
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'modified': obj['LastModified']
                })
                total_size += obj['Size']
            
            return {
                'file_count': len(files),
                'total_size_mb': total_size / (1024 * 1024),
                'files': files
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def generate_report(self, bucket: str = None) -> Dict:
        """Generate comprehensive training report"""
        instances = self.get_training_instances()
        costs = self.estimate_costs(instances)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'instances': instances,
            'costs': costs,
            'summary': {
                'total_instances': len(instances),
                'estimated_total_cost': costs['total_cost']
            }
        }
        
        if bucket:
            s3_info = self.check_s3_training_outputs(bucket)
            report['s3_outputs'] = s3_info
        
        return report
    
    def print_status_report(self, bucket: str = None):
        """Print formatted status report to console"""
        report = self.generate_report(bucket)
        
        print("\n" + "="*60)
        print("HPE TRAINING STATUS REPORT")
        print("="*60)
        print(f"Generated: {report['timestamp']}")
        print()
        
        print(f"Active Instances: {report['summary']['total_instances']}")
        print(f"Estimated Total Cost: ${report['summary']['estimated_total_cost']:.2f}")
        print()
        
        if report['instances']:
            print("INSTANCE DETAILS:")
            print("-" * 40)
            for instance in report['instances']:
                print(f"ID: {instance['instance_id']}")
                print(f"Type: {instance['instance_type']}")
                print(f"State: {instance['state']}")
                print(f"Public IP: {instance['public_ip']}")
                print(f"Launch Time: {instance['launch_time']}")
                
                # Cost info
                costs = report['costs']['instance_costs'].get(instance['instance_id'], {})
                if costs:
                    print(f"Uptime: {costs['uptime_hours']:.2f} hours")
                    print(f"Cost: ${costs['total_cost']:.2f}")
                
                print("-" * 40)
        
        if 's3_outputs' in report and 'error' not in report['s3_outputs']:
            s3_info = report['s3_outputs']
            print(f"\nS3 OUTPUTS:")
            print(f"Files: {s3_info['file_count']}")
            print(f"Total Size: {s3_info['total_size_mb']:.2f} MB")
            print()
            
            if s3_info['files']:
                print("Recent Files:")
                for file_info in sorted(s3_info['files'], 
                                      key=lambda x: x['modified'], reverse=True)[:5]:
                    print(f"  {file_info['key']} ({file_info['size']} bytes)")
        
        print("="*60)

# Flask Dashboard (optional)
if DASHBOARD_AVAILABLE:
    app = Flask(__name__)
    monitor = None
    
    @app.route('/')
    def dashboard():
        return render_template('dashboard.html')
    
    @app.route('/api/status')
    def api_status():
        report = monitor.generate_report()
        return jsonify(report)
    
    @app.route('/api/metrics/<instance_id>')
    def api_metrics(instance_id):
        metrics = monitor.get_training_metrics(instance_id)
        return jsonify(metrics)
    
    def create_dashboard_template():
        """Create simple HTML template for dashboard"""
        template_dir = Path('templates')
        template_dir.mkdir(exist_ok=True)
        
        template_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>HPE Training Monitor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }
        .status { padding: 5px 10px; border-radius: 3px; color: white; }
        .running { background-color: green; }
        .pending { background-color: orange; }
        .terminated { background-color: red; }
    </style>
</head>
<body>
    <h1>HPE Training Monitor</h1>
    
    <div id="status-summary"></div>
    <div id="instance-list"></div>
    <div id="cost-chart"></div>
    <div id="metrics-chart"></div>
    
    <script>
        async function updateDashboard() {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            // Update summary
            document.getElementById('status-summary').innerHTML = `
                <div class="metric-card">
                    <h3>Summary</h3>
                    <p>Active Instances: ${data.summary.total_instances}</p>
                    <p>Estimated Cost: $${data.summary.estimated_total_cost.toFixed(2)}</p>
                </div>
            `;
            
            // Update instance list
            let instancesHtml = '<h3>Instances</h3>';
            data.instances.forEach(instance => {
                instancesHtml += `
                    <div class="metric-card">
                        <h4>${instance.instance_id}</h4>
                        <p>Type: ${instance.instance_type}</p>
                        <p>State: <span class="status ${instance.state}">${instance.state}</span></p>
                        <p>Public IP: ${instance.public_ip}</p>
                    </div>
                `;
            });
            document.getElementById('instance-list').innerHTML = instancesHtml;
        }
        
        // Update every 30 seconds
        setInterval(updateDashboard, 30000);
        updateDashboard();
    </script>
</body>
</html>
        '''
        
        with open(template_dir / 'dashboard.html', 'w') as f:
            f.write(template_content)

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Monitor HPE training on AWS")
    parser.add_argument('--region', default='us-east-2', help='AWS region')
    parser.add_argument('--bucket', help='S3 bucket to check for outputs')
    parser.add_argument('--dashboard', action='store_true', help='Start web dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port')
    parser.add_argument('--continuous', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=60, help='Update interval for continuous mode (seconds)')
    
    args = parser.parse_args()
    
    global monitor
    monitor = HPETrainingMonitor(region=args.region)
    
    if args.dashboard and DASHBOARD_AVAILABLE:
        # Start web dashboard
        create_dashboard_template()
        print(f"Starting web dashboard on http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=False)
    
    elif args.continuous:
        # Continuous monitoring mode
        print(f"Starting continuous monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                monitor.print_status_report(args.bucket)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    
    else:
        # Single report
        monitor.print_status_report(args.bucket)

if __name__ == "__main__":
    main()
