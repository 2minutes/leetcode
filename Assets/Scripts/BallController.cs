using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class BallController : MonoBehaviour
{
    private Rigidbody player_rg;
    private Rigidbody ball_rg;
    private bool isPickUp;

    private Vector3 carry_offset;

    public Button button;
    public float shot_speed;

    // Start is called before the first frame update
    void Start()
    {
        isPickUp = false;
        carry_offset = new Vector3(1.5f, 3.0f, 1.5f);
        button.onClick.AddListener(TaskOnClick);
        ball_rg = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        if (isPickUp)
        {
            transform.position = player_rg.position + carry_offset;
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.tag.CompareTo("Player") == 0 && !isPickUp)
        {
            player_rg = collision.collider.attachedRigidbody;
            isPickUp = true;
        }
    }

    void TaskOnClick()
    {
        if (isPickUp)
        {
            
            transform.position = new Vector3(transform.position.x, 0.5f, transform.position.z);
            ball_rg.velocity = player_rg.transform.forward * shot_speed;
            isPickUp = false;
        }
    }

}
