using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    private Rigidbody rg;
    public float speed;

    public Joystick joystick;

    // Start is called before the first frame update
    void Start()
    {
        rg = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        float MoveHorizentally = joystick.Horizontal;
        float MoveVertically = joystick.Vertical;
        if(MoveHorizentally == 0.0f && MoveVertically == 0.0f)
        {
            rg.velocity = rg.velocity / 1.1f;
        }
        else
        {
            Vector3 movement = new Vector3(MoveHorizentally, 0.0f, MoveVertically);
            rg.AddForce(movement * speed);
            transform.rotation = Quaternion.LookRotation(movement);
        }
        

    }

}
