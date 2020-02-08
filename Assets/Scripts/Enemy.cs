using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Enemy : MonoBehaviour
{
    private float timer;
    private bool flag;
    // Start is called before the first frame update
    void Start()
    {
        flag = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (flag)
        {
            if(timer > 1)
            {
                Destroy(this.gameObject);
            }
            else
            {
                timer += 0.03f;
            }
            
        }
        

    }

    void OnTriggerEnter(Collider other) {
        if ((other.tag.CompareTo("Player") == 0 || other.tag.CompareTo("Bound") == 0)&& flag == false)
        {
            flag = true;
            timer = 0;
        }
    }
}
