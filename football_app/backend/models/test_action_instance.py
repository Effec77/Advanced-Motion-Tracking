"""
Test script for ActionInstance data structure and folder ingestion logic.

This script demonstrates how to use the ActionInstance loader with the 3dsp dataset.
"""

from pathlib import Path
from action_instance import ActionInstance, ActionInstanceLoader, CameraView, MediaType


def test_single_action_instance():
    """Test loading a single ActionInstance from a folder"""
    print("=" * 80)
    print("Test 1: Loading Single ActionInstance")
    print("=" * 80)
    
    # Load a single action from the test dataset
    test_folder = Path("../../../3dsp/test/00001")
    
    try:
        action = ActionInstanceLoader.load_from_folder(test_folder)
        
        print(f"\nLoaded: {action}")
        print(f"Instance ID: {action.instance_id}")
        print(f"Number of camera views: {action.num_views()}")
        print(f"Source folder: {action.source_folder}")
        
        print("\nMetadata:")
        for key, value in action.metadata.items():
            print(f"  [{key}]")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"    {value}")
        
        print("\nCamera Views:")
        for view in action.camera_views:
            print(f"  - {view}")
            print(f"    Media Type: {view.media_type.value}")
            print(f"    Path: {view.media_path}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_dataset_loading():
    """Test loading multiple ActionInstances from a dataset"""
    print("\n" + "=" * 80)
    print("Test 2: Loading Multiple ActionInstances from Dataset")
    print("=" * 80)
    
    # Load all test actions
    dataset_path = Path("../../../3dsp/test")
    
    try:
        actions = ActionInstanceLoader.load_from_dataset(dataset_path, recursive=False)
        
        print(f"\nLoaded {len(actions)} ActionInstances from {dataset_path}")
        
        print("\nSummary:")
        for action in actions[:5]:  # Show first 5
            print(f"  - {action.instance_id}: {action.num_views()} view(s)")
        
        if len(actions) > 5:
            print(f"  ... and {len(actions) - 5} more")
        
        # Statistics
        total_views = sum(action.num_views() for action in actions)
        avg_views = total_views / len(actions) if actions else 0
        
        print(f"\nStatistics:")
        print(f"  Total ActionInstances: {len(actions)}")
        print(f"  Total Camera Views: {total_views}")
        print(f"  Average Views per Action: {avg_views:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_multi_view_scenario():
    """Test a hypothetical multi-view scenario"""
    print("\n" + "=" * 80)
    print("Test 3: Multi-View ActionInstance (Hypothetical)")
    print("=" * 80)
    
    # Create a hypothetical multi-view ActionInstance
    action = ActionInstance(
        instance_id="multi_view_kick_001",
        metadata={
            "sport": "football",
            "action_type": "kick",
            "duration_seconds": 3.5
        }
    )
    
    # Simulate multiple camera views
    hypothetical_views = [
        ("cam_front", Path("/data/event_001/cam_front.mp4"), MediaType.VIDEO),
        ("cam_side", Path("/data/event_001/cam_side.mp4"), MediaType.VIDEO),
        ("cam_back", Path("/data/event_001/cam_back.mp4"), MediaType.VIDEO),
        ("cam_overhead", Path("/data/event_001/cam_overhead.mp4"), MediaType.VIDEO),
    ]
    
    print(f"\nCreated ActionInstance: {action.instance_id}")
    print(f"Metadata: {action.metadata}")
    print(f"\nSimulated Camera Views:")
    
    for view_id, media_path, media_type in hypothetical_views:
        print(f"  - View ID: {view_id}")
        print(f"    Media Type: {media_type.value}")
        print(f"    Path: {media_path}")
    
    print("\nKey Principle:")
    print("  Multiple videos ≠ multiple actions")
    print("  Multiple videos = multiple observations of the same action")
    print("  Physical reality does not change with camera angle.")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ActionInstance Data Structure and Folder Ingestion Tests")
    print("=" * 80)
    
    results = []
    
    # Test 1: Single ActionInstance
    results.append(("Single ActionInstance", test_single_action_instance()))
    
    # Test 2: Dataset Loading
    results.append(("Dataset Loading", test_dataset_loading()))
    
    # Test 3: Multi-View Scenario
    results.append(("Multi-View Scenario", test_multi_view_scenario()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))


if __name__ == "__main__":
    main()
