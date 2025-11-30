#!/usr/bin/env python3
"""
Phase 5: Monitoring and Analytics for Human Feedback Collection
Real-time monitoring, analysis, and reporting for the closed-loop feedback system.
"""

import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class Phase5Monitor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_aws_clients()
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_aws_clients(self):
        """Initialize AWS service clients."""
        region = self.config['aws']['region']
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sqs = boto3.client('sqs', region_name=region)
        
    def _setup_logging(self):
        """Configure logging for the monitor."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_real_time_metrics(self) -> Dict:
        """Get real-time metrics from CloudWatch and DynamoDB."""
        self.logger.info("Collecting real-time metrics...")
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'feedback_collection': self._get_feedback_metrics(),
            'image_generation': self._get_generation_metrics(),
            'system_performance': self._get_system_metrics(),
            'cost_analysis': self._get_cost_metrics()
        }
        
        return metrics
    
    def _get_feedback_metrics(self) -> Dict:
        """Get feedback collection metrics."""
        try:
            table_name = self.config.get('feedback_table', 'vlr-feedback-table')
            table = self.dynamodb.Table(table_name)
            
            # Get recent feedback entries
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            response = table.scan(
                FilterExpression='#ts BETWEEN :start AND :end',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={
                    ':start': start_time.isoformat(),
                    ':end': end_time.isoformat()
                }
            )
            
            feedback_data = response['Items']
            
            # Calculate metrics
            total_responses = len([item for item in feedback_data if 'selected_image_index' in item])
            total_skips = len([item for item in feedback_data if item.get('action') == 'skip'])
            
            # User engagement metrics
            unique_users = len(set(item['user_id'] for item in feedback_data))
            
            # Response time analysis
            response_times = [
                item.get('response_time_ms', 0) for item in feedback_data
                if 'response_time_ms' in item and item['response_time_ms'] > 0
            ]
            
            avg_response_time = np.mean(response_times) if response_times else 0
            median_response_time = np.median(response_times) if response_times else 0
            
            # Completion rate
            total_sessions = len(set(item.get('session_id') for item in feedback_data if item.get('session_id')))
            completion_rate = total_responses / (total_responses + total_skips) if (total_responses + total_skips) > 0 else 0
            
            return {
                'total_responses_24h': total_responses,
                'total_skips_24h': total_skips,
                'unique_users_24h': unique_users,
                'completion_rate': completion_rate,
                'avg_response_time_ms': avg_response_time,
                'median_response_time_ms': median_response_time,
                'total_sessions': total_sessions,
                'responses_per_session': total_responses / total_sessions if total_sessions > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting feedback metrics: {e}")
            return {}
    
    def _get_generation_metrics(self) -> Dict:
        """Get image generation metrics."""
        try:
            # List recent image generation outputs
            bucket_name = self.config.get('generated_images_bucket', '')
            if not bucket_name:
                return {}
            
            # Get objects from last 24 hours
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            response = self.s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix='generated_images/'
            )
            
            recent_objects = []
            for obj in response.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) >= start_time:
                    recent_objects.append(obj)
            
            # Count images by type
            images_generated = len([obj for obj in recent_objects if obj['Key'].endswith('.png')])
            metadata_files = len([obj for obj in recent_objects if obj['Key'].endswith('.json')])
            
            # Estimate generation rate
            total_size_mb = sum(obj['Size'] for obj in recent_objects) / (1024 * 1024)
            
            return {
                'images_generated_24h': images_generated,
                'metadata_files_24h': metadata_files,
                'total_storage_mb': total_size_mb,
                'avg_image_size_mb': total_size_mb / images_generated if images_generated > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting generation metrics: {e}")
            return {}
    
    def _get_system_metrics(self) -> Dict:
        """Get system performance metrics from CloudWatch."""
        try:
            # Get EC2 instance metrics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            # CPU utilization
            cpu_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                StartTime=start_time,
                EndTime=end_time,
                Period=300,  # 5 minutes
                Statistics=['Average']
            )
            
            # Lambda metrics
            lambda_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Duration',
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )
            
            avg_cpu = np.mean([point['Average'] for point in cpu_response['Datapoints']]) if cpu_response['Datapoints'] else 0
            avg_lambda_duration = np.mean([point['Average'] for point in lambda_response['Datapoints']]) if lambda_response['Datapoints'] else 0
            
            return {
                'avg_cpu_utilization': avg_cpu,
                'avg_lambda_duration_ms': avg_lambda_duration,
                'ec2_datapoints': len(cpu_response['Datapoints']),
                'lambda_datapoints': len(lambda_response['Datapoints'])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def _get_cost_metrics(self) -> Dict:
        """Get cost analysis metrics."""
        try:
            # Get billing metrics from CloudWatch
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=1)
            
            cost_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/Billing',
                MetricName='EstimatedCharges',
                Dimensions=[{'Name': 'Currency', 'Value': 'USD'}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=['Maximum']
            )
            
            current_charges = cost_response['Datapoints'][-1]['Maximum'] if cost_response['Datapoints'] else 0
            
            # Estimate hourly cost
            hourly_costs = []
            for i in range(1, len(cost_response['Datapoints'])):
                cost_diff = cost_response['Datapoints'][i]['Maximum'] - cost_response['Datapoints'][i-1]['Maximum']
                hourly_costs.append(max(0, cost_diff))
            
            avg_hourly_cost = np.mean(hourly_costs) if hourly_costs else 0
            
            return {
                'current_estimated_charges_usd': current_charges,
                'avg_hourly_cost_usd': avg_hourly_cost,
                'projected_daily_cost_usd': avg_hourly_cost * 24,
                'cost_datapoints': len(cost_response['Datapoints'])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cost metrics: {e}")
            return {}
    
    def analyze_feedback_patterns(self, days_back: int = 7) -> Dict:
        """Analyze human feedback patterns and quality."""
        self.logger.info(f"Analyzing feedback patterns for the last {days_back} days...")
        
        try:
            table_name = self.config.get('feedback_table', 'vlr-feedback-table')
            table = self.dynamodb.Table(table_name)
            
            # Get feedback data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            
            response = table.scan(
                FilterExpression='#ts BETWEEN :start AND :end AND attribute_exists(selected_image_index)',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={
                    ':start': start_time.isoformat(),
                    ':end': end_time.isoformat()
                }
            )
            
            feedback_data = response['Items']
            df = pd.DataFrame(feedback_data)
            
            if df.empty:
                return {'error': 'No feedback data available'}
            
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['selected_image_index'] = df['selected_image_index'].astype(int)
            
            analysis = {
                'data_summary': self._analyze_data_summary(df),
                'user_behavior': self._analyze_user_behavior(df),
                'response_patterns': self._analyze_response_patterns(df),
                'quality_metrics': self._analyze_quality_metrics(df),
                'temporal_patterns': self._analyze_temporal_patterns(df)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing feedback patterns: {e}")
            return {'error': str(e)}
    
    def _analyze_data_summary(self, df: pd.DataFrame) -> Dict:
        """Analyze basic data summary statistics."""
        return {
            'total_responses': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_triplets': df['triplet_id'].nunique(),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'responses_per_user': {
                'mean': df['user_id'].value_counts().mean(),
                'median': df['user_id'].value_counts().median(),
                'std': df['user_id'].value_counts().std()
            }
        }
    
    def _analyze_user_behavior(self, df: pd.DataFrame) -> Dict:
        """Analyze user behavior patterns."""
        # Response time analysis
        response_times = df['response_time_ms'].dropna()
        
        # User engagement metrics
        user_sessions = df.groupby('user_id').agg({
            'timestamp': ['count', 'min', 'max'],
            'response_time_ms': 'mean'
        }).reset_index()
        
        user_sessions.columns = ['user_id', 'total_responses', 'first_response', 'last_response', 'avg_response_time']
        user_sessions['session_duration_hours'] = (user_sessions['last_response'] - user_sessions['first_response']).dt.total_seconds() / 3600
        
        return {
            'response_time_stats': {
                'mean_ms': response_times.mean(),
                'median_ms': response_times.median(),
                'std_ms': response_times.std(),
                'min_ms': response_times.min(),
                'max_ms': response_times.max()
            },
            'user_engagement': {
                'avg_responses_per_user': user_sessions['total_responses'].mean(),
                'avg_session_duration_hours': user_sessions['session_duration_hours'].mean(),
                'user_retention_rate': (user_sessions['total_responses'] > 1).mean()
            }
        }
    
    def _analyze_response_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze response selection patterns."""
        # Choice distribution
        choice_dist = df['selected_image_index'].value_counts().sort_index()
        
        # Agreement analysis by triplet
        triplet_agreement = df.groupby('triplet_id')['selected_image_index'].agg(['count', lambda x: x.value_counts().max() / len(x)])
        triplet_agreement.columns = ['response_count', 'agreement_rate']
        
        return {
            'choice_distribution': choice_dist.to_dict(),
            'agreement_analysis': {
                'avg_agreement_rate': triplet_agreement['agreement_rate'].mean(),
                'median_agreement_rate': triplet_agreement['agreement_rate'].median(),
                'high_agreement_triplets': (triplet_agreement['agreement_rate'] > 0.8).sum(),
                'low_agreement_triplets': (triplet_agreement['agreement_rate'] < 0.4).sum()
            },
            'response_consistency': {
                'triplets_with_multiple_responses': (triplet_agreement['response_count'] > 1).sum(),
                'avg_responses_per_triplet': triplet_agreement['response_count'].mean()
            }
        }
    
    def _analyze_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Analyze data quality metrics."""
        # Identify potential outliers
        response_times = df['response_time_ms'].dropna()
        q1 = response_times.quantile(0.25)
        q3 = response_times.quantile(0.75)
        iqr = q3 - q1
        
        outlier_threshold_low = q1 - 1.5 * iqr
        outlier_threshold_high = q3 + 1.5 * iqr
        
        time_outliers = response_times[(response_times < outlier_threshold_low) | (response_times > outlier_threshold_high)]
        
        return {
            'data_quality': {
                'missing_response_times': df['response_time_ms'].isna().sum(),
                'missing_session_ids': df['session_id'].isna().sum(),
                'time_outliers_count': len(time_outliers),
                'time_outlier_percentage': len(time_outliers) / len(response_times) * 100
            },
            'response_validity': {
                'valid_image_indices': ((df['selected_image_index'] >= 0) & (df['selected_image_index'] <= 2)).sum(),
                'invalid_responses': ((df['selected_image_index'] < 0) | (df['selected_image_index'] > 2)).sum()
            }
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in responses."""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['date'] = df['timestamp'].dt.date
        
        hourly_patterns = df.groupby('hour').size()
        daily_patterns = df.groupby('day_of_week').size()
        daily_volume = df.groupby('date').size()
        
        return {
            'hourly_distribution': hourly_patterns.to_dict(),
            'daily_distribution': daily_patterns.to_dict(),
            'peak_hours': hourly_patterns.idxmax(),
            'peak_day': daily_patterns.idxmax(),
            'daily_volume_trend': {
                'mean_responses_per_day': daily_volume.mean(),
                'std_responses_per_day': daily_volume.std(),
                'max_responses_per_day': daily_volume.max()
            }
        }
    
    def create_monitoring_dashboard(self, output_file: str = 'phase5_dashboard.html'):
        """Create an interactive monitoring dashboard."""
        self.logger.info("Creating monitoring dashboard...")
        
        # Get real-time metrics
        metrics = self.get_real_time_metrics()
        
        # Get feedback analysis
        feedback_analysis = self.analyze_feedback_patterns()
        
        # Create dashboard
        dashboard_html = self._generate_dashboard_html(metrics, feedback_analysis)
        
        with open(output_file, 'w') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Dashboard created: {output_file}")
    
    def _generate_dashboard_html(self, metrics: Dict, analysis: Dict) -> str:
        """Generate HTML dashboard."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phase 5: Human Feedback Monitoring Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-card { 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 20px; 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    min-width: 200px;
                    background: #f9f9f9;
                }
                .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
                .metric-label { font-size: 0.9em; color: #666; }
                .section { margin: 30px 0; }
                .plot-container { width: 100%; height: 400px; margin: 20px 0; }
                .header { text-align: center; margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Phase 5: Human Feedback Monitoring Dashboard</h1>
                <p>Last Updated: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Real-Time Metrics</h2>
                {metric_cards}
            </div>
            
            <div class="section">
                <h2>Feedback Analysis</h2>
                <div id="feedback-plot" class="plot-container"></div>
            </div>
            
            <div class="section">
                <h2>System Performance</h2>
                <div id="performance-plot" class="plot-container"></div>
            </div>
            
            <script>
                // Add interactive plots here
                {plot_scripts}
            </script>
        </body>
        </html>
        """.format(
            timestamp=metrics.get('timestamp', ''),
            metric_cards=self._generate_metric_cards(metrics),
            plot_scripts=self._generate_plot_scripts(metrics, analysis)
        )
        
        return html_template
    
    def _generate_metric_cards(self, metrics: Dict) -> str:
        """Generate metric cards HTML."""
        cards_html = ""
        
        feedback_metrics = metrics.get('feedback_collection', {})
        generation_metrics = metrics.get('image_generation', {})
        cost_metrics = metrics.get('cost_analysis', {})
        
        key_metrics = [
            ('Responses (24h)', feedback_metrics.get('total_responses_24h', 0)),
            ('Unique Users (24h)', feedback_metrics.get('unique_users_24h', 0)),
            ('Completion Rate', f"{feedback_metrics.get('completion_rate', 0):.1%}"),
            ('Images Generated (24h)', generation_metrics.get('images_generated_24h', 0)),
            ('Est. Cost (USD)', f"${cost_metrics.get('current_estimated_charges_usd', 0):.2f}"),
            ('Avg Response Time (ms)', f"{feedback_metrics.get('avg_response_time_ms', 0):.0f}")
        ]
        
        for label, value in key_metrics:
            cards_html += f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """
        
        return cards_html
    
    def _generate_plot_scripts(self, metrics: Dict, analysis: Dict) -> str:
        """Generate JavaScript for interactive plots."""
        # This is a simplified version - you would add actual Plotly.js code here
        return """
        // Placeholder for interactive plots
        console.log('Dashboard loaded with metrics:', arguments);
        """
    
    def generate_report(self, output_file: str = 'phase5_report.pdf'):
        """Generate a comprehensive PDF report."""
        self.logger.info("Generating comprehensive report...")
        
        # This would use libraries like reportlab or matplotlib to create a PDF
        # For now, we'll create a detailed text report
        
        metrics = self.get_real_time_metrics()
        analysis = self.analyze_feedback_patterns()
        
        report_content = self._format_report_content(metrics, analysis)
        
        # Save as text file (could be enhanced to PDF)
        txt_file = output_file.replace('.pdf', '.txt')
        with open(txt_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Report saved: {txt_file}")
        
    def _format_report_content(self, metrics: Dict, analysis: Dict) -> str:
        """Format report content as text."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        content = f"""
PHASE 5: HUMAN FEEDBACK COLLECTION REPORT
=========================================
Generated: {timestamp}

EXECUTIVE SUMMARY
-----------------
This report provides a comprehensive analysis of the human feedback collection
system for the VLR project Phase 5.

REAL-TIME METRICS
-----------------
{json.dumps(metrics, indent=2)}

FEEDBACK ANALYSIS
-----------------
{json.dumps(analysis, indent=2)}

RECOMMENDATIONS
---------------
Based on the analysis above, here are key recommendations:

1. Monitor completion rates and adjust interface if rates drop below 80%
2. Optimize response time by targeting 2-5 second average response times
3. Ensure adequate user engagement with multiple responses per triplet
4. Monitor costs and adjust instance types if necessary

NEXT STEPS
----------
1. Continue monitoring system performance
2. Analyze human feedback quality and consistency
3. Prepare for closed-loop model retraining
4. Scale system based on feedback collection needs
        """
        
        return content

def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 5 Monitoring and Analytics')
    parser.add_argument('--config', default='config/phase5_config.yaml', help='Config file path')
    parser.add_argument('--dashboard', action='store_true', help='Create monitoring dashboard')
    parser.add_argument('--report', action='store_true', help='Generate PDF report')
    parser.add_argument('--real-time', action='store_true', help='Show real-time metrics')
    parser.add_argument('--analyze', action='store_true', help='Run feedback pattern analysis')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = Phase5Monitor(args.config)
    
    if args.real_time:
        metrics = monitor.get_real_time_metrics()
        print(json.dumps(metrics, indent=2))
    
    if args.analyze:
        analysis = monitor.analyze_feedback_patterns()
        print(json.dumps(analysis, indent=2))
    
    if args.dashboard:
        monitor.create_monitoring_dashboard()
        print("Dashboard created: phase5_dashboard.html")
    
    if args.report:
        monitor.generate_report()
        print("Report generated: phase5_report.txt")

if __name__ == "__main__":
    main()
